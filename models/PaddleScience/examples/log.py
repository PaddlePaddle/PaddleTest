"""
mapping tool
"""

import sys
import re
import numpy as np
import matplotlib.pyplot as plt


def get_curve_data(file_path):
    """
    get data
    """
    with open(file_path, "r") as f:
        data = f.readlines()
        curve_data = []
        for item in data:
            tmp = []
            epoch = re.compile(r"(?<=num_epoch:..)\d*")
            batch = re.compile(r"(?<=num_batch:..)\d*")
            loss_pattern = re.compile(r"(?<=loss:..)\d+\.?\d*")
            eq_loss_pattern = re.compile(r"(?<=eq_loss:..)\d+\.?\d*")
            bc_loss_pattern = re.compile(r"(?<=bc_loss:..)\d+\.?\d*")

            # ic_loss_pattern = re.compile(r'(?<=bc_loss:..)\d+\.?\d*')

            if re.findall(epoch, item):
                if re.findall(epoch, item)[0]:
                    # print(re.findall(epoch, item)[0])
                    num_epoch = int(re.findall(epoch, item)[0])
                    tmp.append(num_epoch)
            if re.findall(batch, item):
                num_batch = int(re.findall(batch, item)[0])
                tmp.append(num_batch)
            if re.findall(loss_pattern, item):
                loss = float(re.findall(loss_pattern, item)[0])
                tmp.append(loss)
            if re.findall(eq_loss_pattern, item):
                eq_loss = float(re.findall(eq_loss_pattern, item)[0])
                tmp.append(eq_loss)
            if re.findall(bc_loss_pattern, item):
                bc_loss = float(re.findall(bc_loss_pattern, item)[0])
                tmp.append(bc_loss)
            if len(tmp) == 5:
                curve_data.append(tmp)

    return np.array(curve_data)


def show_curve(data):
    """
    mapping
    """
    x = data[:, 0]
    loss = data[:, 2]
    eq_loss = data[:, 3]
    bc_loss = data[:, 4]
    plt.figure(figsize=(16, 12), dpi=400)
    plt.plot(x, loss, "ro-", label="loss")
    plt.plot(x, eq_loss, "b--", label="eq_loss")
    plt.plot(x, bc_loss, "g+-", label="bc_loss")
    plt.legend()
    plt.savefig("%s.jpg" % sys.argv[1].split(".")[0])


def main(file_path):
    """
    main function
    """
    data = get_curve_data(file_path)
    show_curve(data)


if __name__ == "__main__":
    file_path = sys.argv[1]
    main(file_path)
