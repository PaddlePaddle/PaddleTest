"""first_order_motion"""
import os
import paddlehub as hub
import paddle

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_first_order_motion_predict():
    """first_order_motion predict"""
    os.system("hub install first_order_motion")
    module = hub.Module(name="first_order_motion")
    module.generate(
        source_image="doc_img.jpeg",
        driving_video="doc_video.mp4",
        ratio=0.4,
        image_size=256,
        output_dir="motion_driving_result",
        filename="result.mp4",
        use_gpu=use_gpu,
    )
    os.system("hub uninstall first_order_motion")
