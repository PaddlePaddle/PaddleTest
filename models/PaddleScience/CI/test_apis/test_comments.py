"""
Test comments
"""
import os


def bracket_check_file(file_path, left_bracket, right_bracket):
    """test"""
    # 正向检查
    s1 = []
    with open(file_path, "r") as f:
        lines = f.read().splitlines()
        for linenum, line in enumerate(lines, start=1):
            for chnum, ch in enumerate(line, start=1):
                if ch == left_bracket:
                    s1.append((ch, linenum, chnum))
                elif ch == right_bracket:
                    if not s1:
                        raise ValueError(f"error at file {file_path}:{linenum}:{chnum}")
                    elif s1[-1][0] != left_bracket:
                        raise ValueError(
                            f"error at file {file_path}:{linenum}:{chnum} which is pair to "
                            f"{file_path}:{s1[-1][1]}:{s1[-1][2]} for bracket: {left_bracket}, {right_bracket}"
                        )
                    else:
                        s1.pop()

    # 反向检查
    s1 = []
    with open(file_path, "r") as f:
        lines = f.read().splitlines()
        lines = lines[::-1]
        for linenum, line in enumerate(lines, start=1):
            line = line[::-1]
            for chnum, ch in enumerate(line, start=1):
                if ch == right_bracket:
                    s1.append((ch, linenum, chnum))
                elif ch == left_bracket:
                    if not s1:
                        raise ValueError(f"error at file {file_path}:{linenum}:{chnum}")
                    elif s1[-1][0] != right_bracket:
                        raise ValueError(
                            f"error at file {file_path}:{linenum}:{chnum} which is pair to "
                            f"{file_path}:{s1[-1][1]}:{s1[-1][2]} for bracket: {left_bracket}, {right_bracket}"
                        )
                    else:
                        s1.pop()


def solve(path: str, whitelist: list = []):

    """test"""

    print(f"solving file {path}")
    if os.path.isdir(path):
        pathes = [o for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))]
        for path_ in pathes:
            solve(os.path.join(path, path_), whitelist)
        pathes = os.listdir(path)
        for path_ in pathes:
            if path_.endswith(".py"):
                if path_ not in whitelist:
                    solve(os.path.join(path, path_), whitelist)
    else:
        if not path.endswith(".py"):
            return
        if path in whitelist:
            return
        bracket_check_file(path, "(", ")")
        bracket_check_file(path, "[", "]")
        bracket_check_file(path, "{", "}")

def get_all_files(folder_path):
    files_list = []  # 保存文件名的列表

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            files_list.append(file_path)

    return files_list

if __name__ == "__main__":
    current_dir = os.path.abspath(os.curdir)
    parent_dir = os.path.dirname(current_dir)
    grandparent_dir = os.path.dirname(parent_dir)
    print(f"grandparent_dir: {grandparent_dir}")
    folder_path = f'{grandparent_dir}/jointContribution/'  # 文件夹路径
    files_list = get_all_files(folder_path)  # 获取文件夹下所有文件
    # print(f"files_list: {files_list}")
    files_list.append("ad.py")
    solve(grandparent_dir, files_list)
