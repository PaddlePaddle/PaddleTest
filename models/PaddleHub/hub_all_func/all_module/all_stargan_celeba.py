"""stargan_celeba"""
import os
import paddlehub as hub
import paddle

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_stargan_celeba_predict():
    """stargan_celeba predict"""
    os.system("hub install stargan_celeba")
    stargan = hub.Module(name="stargan_celeba")
    test_img_path = ["doc_img.jpeg"]
    trans_attr = ["Blond_Hair"]
    # set input dict
    input_dict = {"image": test_img_path, "style": trans_attr}
    # execute predict and print the result
    results = stargan.generate(data=input_dict)
    print(results)
    os.system("hub uninstall stargan_celeba")
