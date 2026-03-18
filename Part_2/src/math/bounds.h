#pragma once

#include "pt.h"
#include "math/vecmath.h"
#include "math/math_utils.h"

namespace pt {

struct alignas(16) Bounds {
    vec3 pMin;
    vec3 pMax;

    PT_HD Bounds() { reset(); }
    PT_HD Bounds(const Bounds& o) : pMin(o.pMin), pMax(o.pMax) {}
    PT_HD Bounds(const vec3& lo, const vec3& hi) : pMin(lo), pMax(hi) {}

    PT_HD void reset() {
        pMin = vec3(float_max(), float_max(), float_max());
        pMax = -pMin;
    }

    PT_HD void expand(const vec3& p) {
        pMin = vec3(pt::min(pMin.x, p.x), pt::min(pMin.y, p.y), pt::min(pMin.z, p.z));
        pMax = vec3(pt::max(pMax.x, p.x), pt::max(pMax.y, p.y), pt::max(pMax.z, p.z));
    }

    PT_HD void expand(const Bounds& b) {
        pMin = vec3(pt::min(pMin.x, b.pMin.x), pt::min(pMin.y, b.pMin.y), pt::min(pMin.z, b.pMin.z));
        pMax = vec3(pt::max(pMax.x, b.pMax.x), pt::max(pMax.y, b.pMax.y), pt::max(pMax.z, b.pMax.z));
    }

    PT_HD float surfaceArea() const {
        if (pMax.x < pMin.x || pMax.y < pMin.y || pMax.z < pMin.z) return 0.0f;
        vec3 d = diagonal();
        return 2.0f * (d.x * d.y + d.x * d.z + d.y * d.z);
    }

    PT_HD vec3 diagonal() const { return pMax - pMin; }

    PT_HD int maxDimension() const {
        vec3 d = diagonal();
        if (d.x > d.y && d.x > d.z) return 0;
        return d.y > d.z ? 1 : 2;
    }

    PT_HD vec3 offset(const vec3& v) const {
        vec3 o = v - pMin;
        vec3 d = diagonal();
        return vec3(d.x > 0 ? o.x / d.x : 0.f,
                    d.y > 0 ? o.y / d.y : 0.f,
                    d.z > 0 ? o.z / d.z : 0.f);
    }

    PT_HD vec3 centroid() const { return 0.5f * pMax + 0.5f * pMin; }

    PT_HD static Bounds merge(const Bounds& a, const Bounds& b) {
        Bounds r(a);
        r.expand(b);
        return r;
    }
};

} // namespace pt
