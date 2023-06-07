"""
Test comments
"""
import os


def bracket_check_file(file_path, left_bracket, right_bracket):
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


def solve(path: str):
    print(f"solving file {path}")
    if os.path.isdir(path):
        pathes = [o for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))]
        for path_ in pathes:
            solve(os.path.join(path, path_))
        pathes = os.listdir(path)
        for path_ in pathes:
            if path_.endswith(".py"):
                solve(os.path.join(path, path_))
    else:
        if not path.endswith(".py"):
            return
        bracket_check_file(path, "(", ")")
        bracket_check_file(path, "[", "]")
        bracket_check_file(path, "{", "}")


if __name__ == "__main__":
    solve("../../../PaddleScience")
