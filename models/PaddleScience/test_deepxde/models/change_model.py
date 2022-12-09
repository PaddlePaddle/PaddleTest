

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

filedir = "../../deepxde/deepxde/model.py"
add_seed(filedir, "\"\"\"paddle\"\"\"", "        self.loss_list = []\n")
add_seed(filedir, "total_loss = paddle.sum(losses)", "            self.loss_list.append(total_loss.item())\n")
print("Change model.py Success")
