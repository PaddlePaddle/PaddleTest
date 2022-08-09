"""humanseg_mobile"""
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


def test_humanseg_mobile_predict():
    """humanseg_mobile predict"""
    os.system("hub install humanseg_mobile")
    human_seg = hub.Module(name="humanseg_mobile")
    im = cv2.imread("doc_img.jpeg")
    # visualization=True可以用于查看人像分割图片效果，可设置为False提升运行速度。
    res = human_seg.segment(images=[im], visualization=True)
    print(res[0]["data"])
    human_seg.video_segment("doc_video.mp4")
    human_seg.save_inference_model("humanseg_mobile_model_save")
    os.system("hub uninstall humanseg_mobile")
