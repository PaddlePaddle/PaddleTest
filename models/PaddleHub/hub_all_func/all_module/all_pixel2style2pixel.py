"""pixel2style2pixel"""
import os
import paddlehub as hub
import paddle

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_pixel2style2pixel_predict():
    """pixel2style2pixel predict"""
    os.system("hub install pixel2style2pixel")
    module = hub.Module(name="pixel2style2pixel")
    input_path = ["face_01.jpeg"]
    # Read from a file
    module.style_transfer(paths=input_path, output_dir="pixel2style2pixel_transfer_result", use_gpu=use_gpu)
    os.system("hub uninstall pixel2style2pixel")
