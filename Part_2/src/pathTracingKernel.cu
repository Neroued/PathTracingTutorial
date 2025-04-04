#include "config.h"

#include <cuda_runtime.h>
#include <driver_types.h>
#include "cuda_fake.h"

#include "math_utils.h"
#include "random_pcg.cuh"
#include "vec3.h"
#include "Ray.h"
#include "Material.h"
#include "SceneData.h"
#include "SceneConstants.cuh"

BEGIN_NAMESPACE_PT


// 半球均匀采样
PT_GPU vec3 sampleHemisphere(unsigned int* state) {
    float z   = rand_pcg(state);
    float r   = max(0.0f, sqrt(1 - z * z));
    float phi = TWO_PI * rand_pcg(state);
    return vec3(r * cos(phi), r * sin(phi), z);
}

// 将向量 v 投影到 N 的法向半球
PT_GPU vec3 toNormalHemisphere(const vec3& v, const vec3& N) {
    vec3 helper(1.0f, 0.0f, 0.0f);
    if (abs(N.x) > 0.999) { helper = vec3(0.0f, 0.0f, 1.0f); }

    vec3 tangent   = normalize(cross(N, helper));
    vec3 bitangent = normalize(cross(N, tangent));
    return v.x * tangent + v.y * bitangent + v.z * N;
}

// 发射从相机位置出发的初始光线
PT_GPU Ray emitRay(unsigned int* state) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 从相机位置出发，看向 z = 1 平面，平面大小为 2 x 2
    float targetX = 2.0f * x / d_sceneConstants.width - 1;
    float targetY = 2.0f * y / d_sceneConstants.height - 1;

    // 添加微小扰动
    targetX += (rand_pcg(state) - 0.5f) / d_sceneConstants.width;
    targetY += (rand_pcg(state) - 0.5f) / d_sceneConstants.height;

    // 矫正长宽比
    targetY /= static_cast<float>(d_sceneConstants.width) / d_sceneConstants.height;

    vec3 target(targetX, targetY, d_sceneConstants.screenZ);
    vec3 d = normalize(target - d_sceneConstants.cameraPos);
    return Ray(d_sceneConstants.cameraPos, d);
}

// 发射从 o 出发，方向为 N 法向半球随机向量的光线
PT_GPU Ray randomRay(const vec3& o, const vec3& N, unsigned int* state) {
    vec3 d = toNormalHemisphere(sampleHemisphere(state), N);
    return Ray(o, d);
}

PT_GPU HitResult nearestHit(const Ray& ray, const SceneData& data) {
    HitResult res;
    HitResult r;

    float nearest = float_max();
    for (int i = 0; i < data.numTriangles; ++i) {
        r = ray.intersectTriangle(data.triangles[i]);
        if (r.isHit && r.distance < nearest) {
            nearest = r.distance;
            res     = r;
        }
    }
    return res;
}

PT_GPU vec3 pathTracing(const SceneData& data, unsigned int* state) {
    vec3 Lo         = {0.0f, 0.0f, 0.0f}; // 最终颜色
    vec3 throughput = {1.0f, 1.0f, 1.0f}; // 累计颜色

    Ray ray = emitRay(state);

    for (int d = 0; d < d_sceneConstants.depth; ++d) {
        // HitResult hit = nearestHit(ray, data);
        HitResult hit = ray.intersectBVH(data.bvhNodes, data.triangles);
        if (!hit.isHit) { break; }

        Material& mat = data.materials[hit.materialID];

        // 命中光源
        if (mat.emissive > 0.0f) {
            Lo += throughput * mat.baseColor * TWO_PI;
            break;
        }

        // 命中物体
        vec3 wi;
        float pdf;
        vec3 brdfFactor;

        float p = rand_pcg(state); // 判断是否发射镜面反射
        if (p < mat.specularRate) {
            // 镜面反射
            wi         = normalize(reflect(ray.direction, hit.normal));
            pdf        = mat.specularRate;
            brdfFactor = mat.baseColor / pdf;
        } else {
            // 漫反射
            wi             = toNormalHemisphere(sampleHemisphere(state), hit.normal);
            // float cosine_o = max(0.0f, dot(-ray.direction, hit.normal));
            float cosine_i = max(0.0f, dot(wi, hit.normal));
            vec3 f_r       = mat.baseColor / PI;
            pdf            = (1.0 - mat.specularRate) / TWO_PI;
            brdfFactor     = f_r * cosine_i / pdf;
        }

        throughput *= brdfFactor;
        ray = Ray(hit.hitPoint, wi);

        // 俄罗斯轮盘
        if (d > 2) {
            float maxComp     = max(max(throughput.x, throughput.y), throughput.z);
            float surviveProb = min(maxComp, 0.95f);
            if (rand_pcg(state) < surviveProb) { break; }
            throughput /= surviveProb;
        }
    }

    // 对结果做一个clamp，防止出现fireflies
    float clampVal = 20.0f;
    Lo.x           = min(Lo.x, clampVal);
    Lo.y           = min(Lo.y, clampVal);
    Lo.z           = min(Lo.z, clampVal);

    return Lo;
}

// -------------------------------------------------------------------------------------------------- //
// --------------------------------------------核心函数----------------------------------------------- //
// -------------------------------------------------------------------------------------------------- //
PT_KERNEL void kernelTest(cudaSurfaceObject_t surface, SceneData data) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= data.width || y >= data.height) return;

    // 初始化随机数 state
    unsigned int state = rand_pcg_init_state(data.frameCount);

    vec3 acc = {0.0f, 0.0f, 0.0f};
    for (int i = 0; i < d_sceneConstants.samplePerFrame; ++i) { acc += pathTracing(data, &state); }
    acc /= d_sceneConstants.samplePerFrame;

    float4 value   = make_float4(acc, 1.0f);
    int byteOffset = x * sizeof(float4);
    surf2Dwrite(value, surface, byteOffset, y);
}

END_NAMESPACE_PT

extern "C" void launchKernel(cudaSurfaceObject_t surface, pt::SceneData data) {
    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridSize((data.width + blockSize.x - 1) / blockSize.x, (data.height + blockSize.y - 1) / blockSize.y);
    pt::kernelTest<<<gridSize, blockSize>>>(surface, data);
    CHECK_LAUNCH_ERROR();
}
