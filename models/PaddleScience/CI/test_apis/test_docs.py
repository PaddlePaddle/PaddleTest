import os
import sys

def check_single_file_linenum_consistent(file: str):
    with open(file) as f:
        lines = f.read().splitlines()
    for i, line in enumerate(lines):
        if (
            ("```" in line)
            and ("linenums=\"" in line)
            and ("--8<--" in lines[i + 1])
        ):
            assert i + 2 < len(lines), (
                f"line {i} is not followed reference block"
            )
            if ":" not in lines[i + 2]:
                continue
            try:
                code_line_beg_expected = line.split("linenums=")
                code_line_beg_expected = code_line_beg_expected[1].split(" ")[0]
                code_line_beg_expected = int(code_line_beg_expected[1:-1]) # remove ""
                location_line = lines[i + 2]
                tmp = (
                    location_line.split(":")
                )
                code_line_beg = tmp[1]
                code_line_beg = int(code_line_beg)
                code_line_end = None
                if len(tmp) > 3:
                    code_line_end = tmp[2]
                    code_line_end = int(code_line_end)

                assert (code_line_end is None) or (code_line_beg <= code_line_end), (
                    f"code line range is not correct: {(code_line_beg, code_line_end)}"
                )
                if code_line_beg_expected != code_line_beg:
                    raise ValueError(
                        f"Error occur in markdown file: {file}:{i}:{i+2} "
                        f"for linenums({code_line_beg_expected}) is not consistent with"
                        f" slice_begin({code_line_beg})"
                    )
            except Exception as e:
                print(f"\n!!! Error occur when checking [{file}:{i}]\n")
                raise e
    print(f"[{file}] check pass...")


def check_linenum_consistent_recursively(path: str):
    if os.path.isdir(path):
        for sub_path in os.listdir(path):
            check_linenum_consistent_recursively(os.path.join(path, sub_path))
    elif path.endswith(".md"):
        check_single_file_linenum_consistent(path)
    return

if __name__ == "__main__":
    check_linenum_consistent_recursively("../../docs")
    