"""ddparser"""
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


def test_ddparser_predict():
    """ddparser"""
    os.system("hub install ddparser")
    # Load ddparser
    module = hub.Module(name="ddparser")
    # String input
    results = module.parse("百度是一家高科技公司")
    print(results)
    # List input
    results = module.parse(["百度是一家高科技公司", "他送了一本书"])
    print(results)
    # Use POS Tag and probability
    module = hub.Module(name="ddparser", prob=True, use_pos=True)
    results = module.parse("百度是一家高科技公司")
    print(results)
    # Visualization mode
    module = hub.Module(name="ddparser", return_visual=True)
    data = module.visualize("百度是一家高科技公司")
    cv2.imwrite("test_ddparser.jpg", data)
    os.system("hub uninstall ddparser")
