
def get_last_epoch_loss(log_file, epoch_num):
    """
    解析训练日志文件，获取最后一个epoch的loss值
    :param log_file: 日志文件路径
    :epoch_num：epoch数量
    :return: 最后一个epoch的loss值，如果没有找到，则返回空字符串
    """
    last_loss = ""
    last_epoch_str = str(epoch_num)+"/"+str(epoch_num)
    with open(log_file, "r") as f:
        for line in reversed(list(f)):
            if last_epoch_str in line and "[Train]" in line and "[Avg]" in line:
                last_loss = line.split("loss: ")[1].split(",")[0]
                break
    return last_loss

if __name__ == "__main__":
    log_file_path = "test_ldc2d_unsteady_Re10.log"
    epoch_num = 20000
    last_loss = get_last_epoch_loss(log_file_path, epoch_num)
    print("最后一个epoch的loss值为：", last_loss)