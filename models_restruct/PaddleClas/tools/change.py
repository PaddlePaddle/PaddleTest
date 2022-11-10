import os
import yaml
import copy

# save_file_path = open("report_linux_cuda102_py37_develop_save.yaml", "w")
# file_path = "report_linux_cuda102_py37_develop.yaml"
# with open(file_path, 'r') as f:
#   line_all = f.readlines()
#   line = [line.strip() for line in line_all]
#   for info in line:
#     if "ppcls" in info:
#         info = info.replace(":", "^")
#         save_file_path.write(info + '\n')
#     else:
#         save_file_path.write(info + '\n')


with open("report_linux_cuda102_py37_develop.yaml", "r") as f:
    content = yaml.load(f, Loader=yaml.FullLoader)

# print('###content', content["ppcls^configs^Cartoonface^ResNet50_icartoon"])
# print('###content', content["ppcls^configs^ppcls^configs^ImageNet^PPHGNet:PPHGNet_tiny"])
# input()
# content_new = copy.deepcopy(content)
# content_new = copy.copy(content)
# content_new = content
# with open("report_linux_cuda102_py37_develop.yaml", "r") as f:
#     content_new = yaml.load(f, Loader=yaml.FullLoader)

i=1
for key, val in content.items():
    i += 1
    # print('###key', key)
    # if "MixNet" in key:
    #     print('###key', key)
    # if "ppcls" in key:

    #     # content[key.replace(":", "^")] = content.pop(key)

    #     # content.update({key.replace(":", "^"): content.pop(key)})

    #     content_new[key.replace(":", "^")]=content_new[key]
    #     del content_new[key]

    # if "MixNet" in key:
    #     print('###key', key.replace(":", "^"))
    #     input()
print(i)
# with open("report_linux_cuda102_py37_develop_save.yaml", "w") as f:
#     yaml.dump(content_new, f)
