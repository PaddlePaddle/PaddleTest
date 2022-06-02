"""FCN_HRNet_W18_Face_Seg"""
import os
import paddlehub as hub
import paddle

import cv2

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_FCN_HRNet_W18_Face_Seg_predict():
    """FCN_HRNet_W18_Face_Seg predict"""
    os.system("hub install FCN_HRNet_W18_Face_Seg")
    model = hub.Module(name="FCN_HRNet_W18_Face_Seg")
    result = model.Segmentation(images=[cv2.imread("doc_img.jpeg")])
    # or
    # result = model.Segmentation(paths=['doc_img.jpeg'])
    print(result)
    os.system("hub uninstall FCN_HRNet_W18_Face_Seg")
