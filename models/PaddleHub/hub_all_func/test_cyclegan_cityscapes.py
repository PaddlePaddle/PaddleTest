"""cyclegan_cityscapes"""
import os
import paddlehub as hub
import paddle

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_cyclegan_cityscapes_predict():
    """cyclegan_cityscapes"""
    os.system("hub install cyclegan_cityscapes")
    cyclegan = hub.Module(name="cyclegan_cityscapes")
    test_img_path = "doc_img.jpeg"
    # set input dict
    input_dict = {"image": [test_img_path]}
    # execute predict and print the result
    results = cyclegan.generate(data=input_dict)
    print(results)
    os.system("hub uninstall cyclegan_cityscapes")
