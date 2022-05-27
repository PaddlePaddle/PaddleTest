"""attgan_celeba"""
import os
import paddlehub as hub
import paddle

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_attgan_celeba_predict():
    """
    attgan_celeba
    """
    os.system("hub install attgan_celeba")
    attgan = hub.Module(name="attgan_celeba")
    test_img_path = ["doc_img.jpeg"]
    trans_attr = ["Bangs"]
    # set input dict
    input_dict = {"image": test_img_path, "style": trans_attr}
    # execute predict and print the result
    results = attgan.generate(data=input_dict)
    print(results)
    os.system("hub uninstall attgan_celeba")
