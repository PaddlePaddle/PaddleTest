import os
import subprocess
import json
import sys

def find_so_files(root_dir):
    """查找所有 .so 文件"""
    so_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".so") or ".so." in filename:
                full_path = os.path.join(dirpath, filename)
                so_files.append(full_path)
    return so_files

def get_so_dependencies_readelf(so_file):
    """使用 readelf 获取 .so 的依赖"""
    try:
        output = subprocess.check_output(['readelf', '-d', so_file], stderr=subprocess.STDOUT, text=True)
        deps = []
        for line in output.splitlines():
            if 'Shared library' in line:
                start = line.find('[')
                end = line.find(']')
                if start != -1 and end != -1:
                    lib_name = line[start+1:end]
                    deps.append(lib_name)
        return deps
    except subprocess.CalledProcessError as e:
        return [f"Error: {e.output.strip()}"]
    
def compare_so_dependency_jsons(json1_path, json2_path):
    """比较两个 .so 静态依赖 JSON 文件的差异"""
    with open(json1_path, 'r') as f1, open(json2_path, 'r') as f2:
        deps1 = json.load(f1)
        deps2 = json.load(f2)

    so_files1 = set(deps1.keys())
    so_files2 = set(deps2.keys())

    only_in_1 = so_files1 - so_files2
    only_in_2 = so_files2 - so_files1
    common_files = so_files1 & so_files2

    diff_results = []

    if only_in_1:
        diff_results.append(f"🔺 只在 {json1_path} 中存在的 .so 文件:\n" + "\n".join(sorted(only_in_1)))
    if only_in_2:
        diff_results.append(f"🔻 只在 {json2_path} 中存在的 .so 文件:\n" + "\n".join(sorted(only_in_2)))

    for so_file in sorted(common_files):
        deps_1 = set(deps1[so_file])
        deps_2 = set(deps2[so_file])
        if deps_1 != deps_2:
            diff_results.append(
                f"🔁 文件依赖不同: {so_file}\n"
                f"  ➤ {json1_path}: {sorted(deps_1)}\n"
                f"  ➤ {json2_path}: {sorted(deps_2)}"
            )

    if not diff_results:
        print("✅ 两个 JSON 文件中的所有 .so 文件及其依赖完全一致。")
    else:
        print("⚠️ 发现差异：\n")
        print("\n\n".join(diff_results))
        sys.exit(1)

def main(target_dir, output_json):
    # target_dir = "paddle"  # TODO: 修改为你自己的路径
    # output_json = "so_dependencies_static.json"

    so_files = find_so_files(target_dir)
    all_deps = {}

    for so_file in so_files:
        deps = get_so_dependencies_readelf(so_file)
        all_deps[so_file] = deps

    with open(output_json, "w") as f:
        json.dump(all_deps, f, indent=4)

    print(f"✅ 静态依赖信息已保存到 {output_json}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract or compare .so static dependencies")
    subparsers = parser.add_subparsers(dest="command")

    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--target_dir", type=str, default="paddle", help="Target directory to search for .so files")
    extract_parser.add_argument("--output_json", type=str, default="so_dependencies_static.json", help="Output JSON file name")

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("json1", type=str, help="First JSON file to compare")
    compare_parser.add_argument("json2", type=str, help="Second JSON file to compare")

    args = parser.parse_args()

    if args.command == "extract":
        main(args.target_dir, args.output_json)
    elif args.command == "compare":
        compare_so_dependency_jsons(args.json1, args.json2)