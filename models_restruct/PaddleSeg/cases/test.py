import os
with open('models_list_seg_fix_input', 'r') as f:
    for line in f:
        temp = line.strip('\n')
        os.system("cp fix_input.yml {}".format(temp))

