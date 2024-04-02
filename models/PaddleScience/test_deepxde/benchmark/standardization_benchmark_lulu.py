import argparse
from configtrans import YamlLoader


flag_LBFGS= False
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
                if old_str == "L-BFGS":
                    if "    " not in line:
                        global flag_LBFGS
                        flag_LBFGS = True
                        line += new_str
                else:
                    line += new_str
                #line += "paddle.seed(1)\n"
                #line += "np.random.seed(1)\n" 
            file_data += line
    with open(file,"w",encoding="utf-8") as f:
        f.write(file_data)

def change_backend(file,backend,flag):
    """
    change models.py backend
    """
    file_data = ""
    if flag==True:
        index = False
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                if index==True:
                    if "# " in line and "Backend jax" not in line:
                        line = line.replace("# ", "")
                    else:
                        index = False
                if backend in line:
                    index = True
                file_data += line
        with open(file,"w",encoding="utf-8") as f:
            f.write(file_data)                
    else:
        index = False 
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                if index==True:
                    if "Backend paddle" not in line:
                        line = "# " + line
                    else:
                        index = False
                if backend in line:
                    index = True
                file_data += line
        with open(file,"w",encoding="utf-8") as f:
            f.write(file_data)

args = parse_args()
yamldir = YamlLoader(args.file)
case_num, py = yamldir.get_case(0)
filedir = py
alter(filedir, "tf", "paddle")
change_backend(filedir, "Backend paddle", True)
change_backend(filedir,"Backend tensorflow.compat.v1", False)
add_seed(filedir, "import deepxde", "import paddle\n")
#add_seed(filedir, "import paddle", "paddle.seed(1)\n")

print("Successfully replaced the backend as paddle!(benchmark)")