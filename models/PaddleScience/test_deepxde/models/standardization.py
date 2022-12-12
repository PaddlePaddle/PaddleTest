import argparse
from configtrans import YamlLoader

def parse_args():
    """
    parse args
    """
    parser = argparse.ArgumentParser(description='ScienceTest')
    parser.add_argument('-f', '--file', help='set yaml file')
    args = parser.parse_args()
    return args
    

def alter(file,old_str,new_str,flag=True, except_str='model.train(0'):
    """
    replaced the backend 
    """
    file_data = ""
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            if flag:
                if old_str in line and new_str not in line and except_str not in line:
                    line = line.replace(old_str,new_str)
            else:
                if old_str in line:
                    line = line.replace(old_str,new_str)
            file_data += line
    with open(file,"w",encoding="utf-8") as f:
        f.write(file_data)


def add_seed(file,old_str,new_str):
    """
    add the seed 
    """
    file_data = ""
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            if old_str in line:
                line += new_str
                #line += "paddle.seed(1)\n"
                #line += "np.random.seed(1)\n" 
            file_data += line
    with open(file,"w",encoding="utf-8") as f:
        f.write(file_data)


args = parse_args()
yamldir = YamlLoader(args.file)
case_num, py = yamldir.get_case(0)
filedir = py
#alter(filedir, "tf", "paddle")
#alter("../../deepxde/deepxde/backend/paddle/tensor.py", "2.3.0", "0.0.0")
#alter(filedir, "model.train(", "model.train(display_every=1,", True, "model.train(0")
alter(filedir, "model.train(", "losshistory, train_state = model.train(")
#alter(filedir, "display_every=1000", "display_every=1")
#alter(filedir, "display_every=500", " ",False)
add_seed(filedir, "import deepxde", "import paddle\n")
#add_seed(filedir, "import paddle", "paddle.seed(1)\n")
add_seed(filedir, "import deepxde", "import numpy as np\n")
add_seed(filedir, "import deepxde", "dde.config.set_random_seed(1)\n")
#add_seed(filedir, "import numpy as np", "np.random.seed(1)\n")
with open(filedir, "a") as f:
    f.write( "result = model.loss_list[:200]\n"
             "np.save('loss.npy',result)\n"
             "np.save('metric.npy',losshistory.metrics_test[-1])\n"
             )
#add_seed(filedir, "model.train(", "result = np.sum(losshistory.loss_train, axis=1)\n")
#add_seed(filedir, "result = np.sum(losshistory.loss_train, axis=1)", "result = result[:200]\n")
#add_seed(filedir, "result = result[:200]", "np.save('loss.npy',result)\n")
#add_seed(filedir, "np.save('loss.npy',result)", "np.save('metric.npy',model.train_state.best_metric)\n")

print("Successfully replaced the backend as paddle!")