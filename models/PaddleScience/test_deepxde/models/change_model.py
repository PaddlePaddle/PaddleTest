

def add_seed(file,num,new_str):
    """
    add the seed 
    """
    file_data = ""
    index = 0
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            index += 1
            if index==num:
                line += new_str
                #line += "paddle.seed(1)\n"
                #line += "np.random.seed(1)\n" 
            file_data += line
    with open(file,"w",encoding="utf-8") as f:
        f.write(file_data)

filedir = "../../deepxde/deepxde/model.py"
add_seed(filedir, 446, "        self.loss_list = []\n")
add_seed(filedir, 500, "            self.loss_list.append(total_loss.item())\n")
add_seed(filedir, 513, "                self.loss_list.append(total_loss.item())\n")
print("Change model.py Success")
