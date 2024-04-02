"""SINet_Portrait_Segmentation"""
import os
import paddle

import cv2
import paddlehub as hub

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_SINet_Portrait_Segmentation_predict():
    """SINet_Portrait_Segmentation predict"""
    os.system("hub install SINet_Portrait_Segmentation")
    model = hub.Module(name="SINet_Portrait_Segmentation")
    result = model.Segmentation(
        images=[cv2.imread("doc_img.jpeg")],
        paths=None,
        batch_size=1,
        output_dir="SINet_Portrait_Segmentation_output",
        visualization=False,
    )
    print(result)
    os.system("hub uninstall SINet_Portrait_Segmentation")
