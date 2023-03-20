import os
with open('PaddleDetection_TIPC_list', 'r') as f:
    for line in f:
        temp = line.strip('\n')
        os.system("cp test_tipc.yml {}".format(temp))

