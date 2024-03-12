# ################################################ Usage ################################################
# # [] means optional argument
# # python deleter.py --path=dir_path --focus=model_dir_name[,another_name]... [--del_empty] [--status]
# #######################################################################################################

# import os, sys, re
# import argparse
# import pathlib


# parser = argparse.ArgumentParser()
# parser.add_argument('--path', type=str)
# parser.add_argument('--focus', type=str)
# parser.add_argument('--del_empty', action="store_true")
# parser.add_argument('--status', action="store_true")
# args = parser.parse_args()

# layer_set = set()
# delete_file = {}

# total_file = 0
# del_file = 0
# can_del_file = 0

# def extract_forward_meta(code):
#     pattern = r"def forward\([\s\S]*?\):"
#     match = re.search(pattern, code)
#     if not match:
#         return ""

#     forward_part = match.group()

#     # 提取 var_x 行
#     var_pattern = r"\s*(var_\d+,\s*#\s*\(shape:.*?\))"
#     parameters = re.findall(var_pattern, forward_part)

#     if parameters:
#         return "\n".join(parameters)
#     else:
#         return ""

# if args.focus is not None:
#     focus = set(os.path.split(name)[-1] for name in args.focus.split(","))
#     def file_under_focus(filename):
#         global focus
#         parts = pathlib.Path(filename).parts
#         for p in parts:
#             if p in focus:
#                 return True
#         return False
# else:
#     file_under_focus = lambda filename: True

# for dirpath, dirnames, filenames in os.walk(args.path):
#     for filename in filenames:
#         if filename.endswith(".py"):
#             total_file += 1
#             filepath = os.path.join(dirpath, filename)
#             with open(filepath, "r") as f:
#                 first_line = f.readline()
#                 code = f.read()
#                 forward_meta = extract_forward_meta(code)
#                 meta_info = first_line + '\n' + forward_meta
#             if meta_info in layer_set:
#                 if meta_info in delete_file:
#                     delete_file[meta_info].append(filepath)
#                 else:
#                     delete_file[meta_info] = [filepath]
#             else:
#                 layer_set.add(meta_info)


# for meta_info, files in delete_file.items():
#     head = f"With Sub Layer: {meta_info}\n"
#     body = ""
#     for filepath in files:
#         can_del_file += 1
#         if file_under_focus(filepath):
#             body += f"    Delete: {filepath}\n"
#             if args.status is not True:
#                 del_file += 1
#                 os.remove(filepath)
#     if body != "":
#         print(head + body)

# if args.del_empty:
#     for dirpath, dirnames, filenames in os.walk(args.path):
#         if not dirnames and not filenames:
#             os.rmdir(dirpath)


# print(f"Total {total_file} sub graphs")
# print(f"Can Delete {can_del_file}, Left {total_file - can_del_file}")
# print(f"Exactly Delete {del_file}, Left {total_file - del_file}")
