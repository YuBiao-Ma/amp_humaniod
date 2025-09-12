import json
import os

def merge_walk_frames(input_files, output_file):
    """
    合并多个walk_*.txt文件的Frames数据
    :param input_files: 输入文件路径列表
    :param output_file: 输出合并后的TXT文件路径
    """
    # 初始化合并后的数据（None表示未读取到第一个有效文件）
    merged_data = None
    total_frames = 0  # 统计总帧数，便于日志输出

    for file_path in input_files:
        # 跳过不存在的文件
        if not os.path.exists(file_path):
            print(f"警告：文件 {file_path} 不存在，已跳过")
            continue

        try:
            # 读取并解析JSON数据
            with open(file_path, 'r', encoding='utf-8') as f:
                # 处理可能的JSON格式问题（如注释、多余逗号）
                try:
                    file_data = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"错误：文件 {file_path} JSON解析失败（{str(e)}），已跳过")
                    continue

            # 验证数据结构（必须包含Frames键）
            if "Frames" not in file_data:
                print(f"警告：文件 {file_path} 缺少Frames字段，已跳过")
                continue

            # 处理第一个有效文件：保留完整顶层配置
            if merged_data is None:
                merged_data = file_data
                total_frames += len(file_data["Frames"])
                print(f"成功读取第一个文件：{file_path}，初始帧数：{len(file_data['Frames'])}")
            # 处理后续文件：仅追加Frames数据
            else:
                frames_to_add = file_data["Frames"]
                merged_data["Frames"].extend(frames_to_add)
                total_frames += len(frames_to_add)
                print(f"成功合并文件：{file_path}，新增帧数：{len(frames_to_add)}，累计总帧数：{total_frames}")

        except Exception as e:
            print(f"意外错误：处理文件 {file_path} 时出错（{str(e)}），已跳过")
            continue

    # 检查是否有有效数据可输出
    if merged_data is None:
        print("错误：未读取到任何有效数据，无法生成输出文件")
        return
    if len(merged_data["Frames"]) == 0:
        print("错误：合并后Frames为空，无法生成输出文件")
        return

    # 写入合并后的TXT文件（JSON格式）
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # indent=2 保留缩进，ensure_ascii=False 支持特殊字符（若有）
            json.dump(merged_data, f, indent=2, ensure_ascii=False)
        print(f"\n合并完成！输出文件：{os.path.abspath(output_file)}")
        print(f"合并后总帧数：{len(merged_data['Frames'])}")
        print(f"保留的顶层配置（来自第一个有效文件）：{list(merged_data.keys())}")
    except Exception as e:
        print(f"错误：写入输出文件 {output_file} 失败（{str(e)}）")


# -------------------------- 配置参数 --------------------------
# 1. 输入文件列表（按优先级排序，优先读取前面的文件作为顶层配置来源）
INPUT_FILES = [
    "walk_5.txt",
    "walk_6.txt",
    "walk_7.txt"
]
# 2. 输出文件路径（可自定义，如"merged_walk_all.txt"）
OUTPUT_FILE = "merged_walk_frames.txt"
# -------------------------------------------------------------

# 执行合并
if __name__ == "__main__":
    merge_walk_frames(INPUT_FILES, OUTPUT_FILE)
