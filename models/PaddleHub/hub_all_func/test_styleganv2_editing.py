"""styleganv2_editing"""
import os
import paddlehub as hub
import paddle

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_styleganv2_editing_predict():
    """styleganv2_editing predict"""
    os.system("hub install styleganv2_editing")
    module = hub.Module(name="styleganv2_editing")
    input_path = ["face_01.jpeg"]
    # Read from a file
    module.generate(
        paths=input_path,
        direction_name="age",
        direction_offset=5,
        output_dir="styleganv2_editing_result",
        use_gpu=use_gpu,
    )
    os.system("hub uninstall styleganv2_editing")
