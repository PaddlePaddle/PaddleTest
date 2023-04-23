def get_last_epoch_loss(log_file, epoch_num):
    """
    解析训练日志文件，获取最后一个epoch的loss值
    :param log_file: 日志文件路径
    :param epoch_num: epoch数量
    :return: 最后一个epoch的loss值，如果没有找到，则返回空字符串
    """

    last_loss = ""  # 最后一个epoch的loss值变量初始化为空字符串
    last_epoch_str = str(epoch_num) + "/" + str(epoch_num)  # 最后一个epoch的字符串表示形式

    with open(log_file, "r") as f:
        # 倒序读取日志文件中的每一行
        for line in reversed(list(f)):
            if last_epoch_str in line and "[Train]" in line and "[Avg]" in line:
                # 如果该行包含是否为最后一个 epoch、是否为 Train 阶段和是否包含 Avg，则解析该行的 loss 值并赋值给 last_loss 变量
                last_loss = line.split("loss: ")[1].split(",")[0]
                break  # 解析到最后一个 epoch 的 loss 值后退出循环

    return last_loss  # 返回最后一个 epoch 的 loss 值，如果没有找到，则返回空字符串


if __name__ == "__main__":
    log_file_path = "test_ldc2d_unsteady_Re10.log"
    epoch_num = 20000
    last_loss = get_last_epoch_loss(log_file_path, epoch_num)
    print("最后一个epoch的loss值为：", last_loss)
