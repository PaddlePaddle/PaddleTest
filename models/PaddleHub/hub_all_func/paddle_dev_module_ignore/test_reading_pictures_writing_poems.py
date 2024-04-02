"""reading_pictures_writing_poems"""
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


def test_reading_pictures_writing_poems_predict():
    """reading_pictures_writing_poems predict"""
    os.system("hub install reading_pictures_writing_poems")
    readingPicturesWritingPoems = hub.Module(name="reading_pictures_writing_poems")
    results = readingPicturesWritingPoems.WritingPoem(image="doc_img.jpeg", use_gpu=use_gpu)
    for result in results:
        print(result)
    os.system("hub uninstall reading_pictures_writing_poems")
