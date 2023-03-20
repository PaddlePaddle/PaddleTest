import os
with open('models_list_detection_no_model', 'r') as f:
    for line in f:
        temp = line.strip('\n')
        os.system("cp test_no_model.yml {}".format(temp))
