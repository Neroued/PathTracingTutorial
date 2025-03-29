#!/usr/bin/env python3
import json
import shlex


def transform_command(command, file_path=""):
    """
    对单条编译命令进行转换：
    1. 如果命令中不包含 nvcc 或 cl.exe，则直接返回原命令。
    2. 如果包含 nvcc，则将 nvcc 的路径替换为指定的 clang++ 路径，
       同时过滤掉 nvcc 专用的参数，保留对 clangd 补全有用的参数。
       同时，对于处理 .cu 文件的 nvcc 命令，添加 -x cuda 和 -D__CUDACC__ 参数。
    3. 如果包含 cl.exe（MSVC 命令），则将编译器更换为 clang++，
       并去除 MSVC 专用参数，同时将需要的参数转换为 clang 支持的格式，
       例如将 /std:c++17 转换为 -std=c++17，将 /O2 或 /Ob2 转换为 -O2，
       同时将 -external:IXXX 转换为 -isystem XXX，其它 -external:* 参数则直接移除。
    """
    command_lower = command.lower()
    # 如果命令中既不包含 nvcc 也不包含 cl.exe，则不处理
    if "nvcc" not in command_lower and "cl.exe" not in command_lower:
        return command

    try:
        # Windows 下使用 posix=False 以便正确处理路径中的反斜杠
        tokens = shlex.split(command, posix=False)
    except Exception as e:
        print("分割命令出错:", e)
        return command

    new_tokens = []
    skip_next = False

    # 针对 nvcc 命令的转换
    if "nvcc" in command_lower:
        # 需要精确删除的参数
        removal_set = {
            "-forward-unknown-to-host-compiler",
            "-D__CUDACC__",
            "/GR",
            "/EHsc",
        }
        # 需要删除的参数前缀
        removal_prefixes = ["-Xcompiler=", "--generate-code=", "--cuda-gpu-arch="]

        for i, token in enumerate(tokens):
            if skip_next:
                skip_next = False
                continue

            token_clean = token.strip('"')

            # 跳过 "-x cu" 参数对
            if token == "-x" and i + 1 < len(tokens) and tokens[i + 1] == "cu":
                skip_next = True
                continue

            # 如果是 nvcc 的路径（首个 token 且包含 nvcc），替换为指定的 clang++ 路径
            if i == 0 and "nvcc" in token.lower():
                new_tokens.append(r"E:\code\cpplibrary\LLVM\bin\clang++.exe")
                continue

            if token_clean in removal_set:
                continue

            if any(token_clean.startswith(prefix) for prefix in removal_prefixes):
                continue

            if token == "-Xcompiler" and i + 1 < len(tokens):
                skip_next = True
                continue

            # 检查 nvcc 中用于优化的参数 -Ob2，替换为 clang++ 识别的 -O2
            if token_clean == "-Ob2":
                new_tokens.append("-O2")
                continue

            new_tokens.append(token)

        # 如果处理的是 .cu 文件，添加 -x cuda 和 -D__CUDACC__ 参数
        if file_path and file_path.lower().endswith(".cu"):
            new_tokens.extend(["-x", "cuda", "-D__CUDACC__"])

        new_command = " ".join(new_tokens)
        return new_command

    # 针对 MSVC (cl.exe) 命令的转换
    elif "cl.exe" in command_lower:
        for i, token in enumerate(tokens):
            if skip_next:
                skip_next = False
                continue

            token_clean = token.strip('"')

            # 替换编译器路径：如果首个 token 包含 cl.exe，则换为 clang++.exe
            if i == 0 and "cl.exe" in token_clean:
                new_tokens.append(r"E:\code\cpplibrary\LLVM\bin\clang++.exe")
                continue

            # 移除 MSVC 专用的参数（精确匹配）
            if token_clean in {
                "/nologo",
                "/TP",
                "/FS",
                "/MD",
                "/GR",
                "/EHsc",
                "-permissive-",
                "/utf-8",
            }:
                continue

            # 移除输出文件相关参数：/Fo 和 /Fd 及其紧跟的参数（若有空格形式）
            if token_clean.startswith("/Fo") or token_clean.startswith("/Fd"):
                continue

            # 移除 /Zc:* 参数（支持 /Zc:__cplusplus 或 -Zc:__cplusplus）
            if token_clean.startswith("/Zc:") or token_clean.startswith("-Zc:"):
                continue

            # 将 MSVC 的 /std:c++17 转换为 clang++ 的 -std=c++17
            if token_clean.startswith("-std:c++17"):
                new_tokens.append("-std=c++17")
                continue

            # 将优化选项 /O2 或 /Ob2 转换为 -O2
            if token_clean in {"/O2", "/Ob2"}:
                new_tokens.append("-O2")
                continue

            # 处理 -external:* 参数
            if token_clean.startswith("-external:"):
                # 如果是 -external:I 参数，则转换为 -isystem
                if token_clean.startswith("-external:I"):
                    path = token_clean[len("-external:I") :]
                    new_tokens.append("-isystem")
                    new_tokens.append(path)
                # 否则，忽略其他 -external:* 参数（例如 -external:W0）
                continue

            # 保留其他参数
            new_tokens.append(token)

        new_command = " ".join(new_tokens)
        return new_command

    else:
        return command


def main():
    # 固定输入输出文件路径
    input_file = r"build\compile_commands.json"
    output_file = r"utils\compile_commands.json"

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            compile_commands = json.load(f)
    except Exception as e:
        print("读取 JSON 文件出错:", e)
        return

    # 遍历每一条编译命令，对 nvcc 或 MSVC 的命令进行转换
    for entry in compile_commands:
        command = entry.get("command", "")
        file_path = entry.get("file", "")
        new_command = transform_command(command, file_path)
        entry["command"] = new_command

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(compile_commands, f, indent=2)
        print(f"file saved to {output_file}")
    except Exception as e:
        print("写入文件出错:", e)


if __name__ == "__main__":
    main()
