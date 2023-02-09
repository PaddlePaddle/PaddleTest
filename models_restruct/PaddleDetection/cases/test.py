import os
with open('models_list_keypoint', 'r') as f:
    for line in f:
        temp = line.strip('\n')
        os.system("cp test_keypoint.yml {}".format(temp))

