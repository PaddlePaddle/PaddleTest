"""stgan_celeba"""
import os
import paddlehub as hub
import paddle

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_stgan_celeba_predict():
    """stgan_celeba predict"""
    os.system("hub install stgan_celeba")
    stgan = hub.Module(name="stgan_celeba")
    test_img_path = ["doc_img.jpeg"]
    org_info = ["Female,Black_Hair"]
    trans_attr = ["Bangs"]
    # set input dict
    input_dict = {"image": test_img_path, "style": trans_attr, "info": org_info}
    # execute predict and print the result
    results = stgan.generate(data=input_dict)
    print(results)
    os.system("hub uninstall stgan_celeba")
