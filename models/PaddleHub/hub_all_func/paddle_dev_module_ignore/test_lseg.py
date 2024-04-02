"""lseg"""
import os
import cv2
import paddlehub as hub
import paddle

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_lseg_predict():
    """lseg predict"""
    os.system("hub install lseg")
    module = hub.Module(name="lseg")
    result = module.segment(
        image=cv2.imread("doc_img.jpeg"),
        labels=["Category 1", "Category 2", "Category n"],
        visualization=True,
        output_dir="lseg_output",
    )
    print(result)
    os.system("hub uninstall lseg")
