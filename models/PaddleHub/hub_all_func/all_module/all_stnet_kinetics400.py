"""stnet_kinetics400"""
import os
import paddlehub as hub
import paddle

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_stnet_kinetics400_predict():
    """stnet_kinetics400"""
    os.system("hub install stnet_kinetics400")
    stnet = hub.Module(name="stnet_kinetics400")
    test_video_path = "doc_video.mp4"
    # set input dict
    input_dict = {"image": [test_video_path]}
    # execute predict and print the result
    results = stnet.video_classification(data=input_dict)
    for result in results:
        print(result)
    os.system("hub uninstall stnet_kinetics400")
