"""human_pose_estimation_resnet50_mpii"""
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


def test_human_pose_estimation_resnet50_mpii_predict():
    """human_pose_estimation_resnet50_mpii"""
    os.system("hub install human_pose_estimation_resnet50_mpii")
    pose_estimation = hub.Module(name="human_pose_estimation_resnet50_mpii")

    result = pose_estimation.keypoint_detection(images=[cv2.imread("doc_img.jpeg")])
    # or
    # result = pose_estimation.keypoint_detection(paths=['doc_img.jpeg'])
    # PaddleHub示例图片下载方法：
    # wget https://paddlehub.bj.bcebos.com/resources/test_image.jpg
    print(result)
    os.system("hub uninstall human_pose_estimation_resnet50_mpii")
