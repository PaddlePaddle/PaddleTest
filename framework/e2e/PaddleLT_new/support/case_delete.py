# import os, sys
# import argparse
# import pathlib


# parser = argparse.ArgumentParser()
# parser.add_argument("--path", type=str)
# parser.add_argument("--focus", type=str)
# parser.add_argument("--del_empty", action="store_true")
# args = parser.parse_args()

# layer_set = set()
# delete_file = {}

# if args.focus is not None:
#     focus = set(os.path.split(name)[-1] for name in args.focus.split(","))

#     def file_under_foces(filename):
#         global focus
#         parts = pathlib.Path(filename).parts
#         for p in parts:
#             if p in focus:
#                 return True
#         return False

# else:
#     file_under_foces = lambda filename: True

# for dirpath, dirnames, filenames in os.walk(args.path):
#     for filename in filenames:
#         if filename.endswith(".py"):
#             filepath = os.path.join(dirpath, filename)
#             with open(filepath, "r") as f:
#                 first_line = f.readline()
#             if first_line in layer_set:
#                 if first_line in delete_file:
#                     delete_file[first_line].append(filepath)
#                 else:
#                     delete_file[first_line] = [filepath]
#             else:
#                 layer_set.add(first_line)


# for first_line, files in delete_file.items():
#     head = f"With Sub Layer: {first_line}\n"
#     body = ""
#     for filepath in files:
#         if file_under_foces(filepath):
#             body += f"    Delete: {filepath}\n"
#             os.remove(filepath)
#     if body != "":
#         print(head + body)


# if args.del_empty:
#     for dirpath, dirnames, filenames in os.walk(args.path):
#         if not files and not dirs:
#             os.remdir(dirpath)
