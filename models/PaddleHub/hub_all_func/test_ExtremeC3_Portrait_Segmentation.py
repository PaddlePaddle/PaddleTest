"""ExtremeC3_Portrait_Segmentation"""
import os
import cv2
import paddle

import paddlehub as hub

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_ExtremeC3_Portrait_Segmentation_predict():
    """ExtremeC3_Portrait_Segmentation"""
    os.system("hub install ExtremeC3_Portrait_Segmentation")
    model = hub.Module(name="ExtremeC3_Portrait_Segmentation")
    result = model.Segmentation(
        images=[cv2.imread("doc_img.jpeg")],
        paths=None,
        batch_size=1,
        output_dir="ExtremeC3_Portrait_Segmentation_output",
        visualization=False,
    )
    print(result)
    os.system("hub uninstall ExtremeC3_Portrait_Segmentation")
