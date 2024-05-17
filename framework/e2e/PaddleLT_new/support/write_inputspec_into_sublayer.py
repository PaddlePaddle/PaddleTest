import os
import glob
import paddle


def insert_before_function(filename, function_name, content_to_insert):
    """
    在py文件某一行前写入字符串
    """
    with open(filename, "r", encoding="utf-8") as file:
        lines = file.readlines()

    inserted = False
    new_lines = []
    for line in lines:
        if "def " + function_name + "(" in line and not inserted:
            new_lines.append(content_to_insert + "\n")
            new_lines.append(line)
            inserted = True
        else:
            new_lines.append(line)

    if not inserted:
        print(f"Function '{function_name}' not found in file '{filename}'.")
        return

    with open(filename, "w", encoding="utf-8") as file:
        file.writelines(new_lines)


inputspec_dir = "inputspec_dy2st"

inputspec_files = glob.glob(os.path.join(f"{inputspec_dir}", "*.inputspec.tensor"))
for file in inputspec_files:
    spec_tuple = paddle.load(file)

    inputspec_str = "\ndef create_inputspec(): \n"
    inputspec_str += "    inputspec = ( \n"
    for spec in spec_tuple:
        inputspec_str += f"        paddle.static.InputSpec(shape={spec.shape}, dtype={spec.dtype}, stop_gradient={spec.stop_gradient}), \n"

    inputspec_str += "    )\n"
    inputspec_str += "    return inputspec\n"

    # layercase^sublayer1000^Clas_cases^CSWinTransformer_CSWinTransformer_base_384^SIR_1.inputspec.tensor
    sublayer_file = file.replace(".inputspec.tensor", "").replace("^", "/").replace(f"{inputspec_dir}/", "") + ".py"
    print(f"开始写入: {sublayer_file}")
    insert_before_function(
        filename=sublayer_file, function_name="create_tensor_inputs", content_to_insert=inputspec_str
    )
