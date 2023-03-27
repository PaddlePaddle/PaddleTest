import os
with open('models_list_seg_no_model', 'r') as f:
    for line in f:
        temp = line.strip('\n')
        os.system("cp no_model.yml {}".format(temp))

