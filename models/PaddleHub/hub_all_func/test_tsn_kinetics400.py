"""tsn_kinetics400"""
import os
import paddlehub as hub
import paddle

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_tsn_kinetics400_predict():
    """tsn_kinetics400 predict"""
    os.system("hub install tsn_kinetics400")
    tsn = hub.Module(name="tsn_kinetics400")
    test_video_path = "doc_video.mp4"
    # set input dict
    input_dict = {"image": [test_video_path]}
    # execute predict and print the result
    results = tsn.video_classification(data=input_dict)
    for result in results:
        print(result)
    os.system("hub uninstall tsn_kinetics400")
