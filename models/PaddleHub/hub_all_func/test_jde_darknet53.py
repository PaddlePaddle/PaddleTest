"""jde_darknet53"""
import os
import paddlehub as hub
import paddle

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_jde_darknet53_predict():
    """jde_darknet53"""
    os.system("hub install jde_darknet53")
    tracker = hub.Module(name="jde_darknet53")
    # Read from a video file
    tracker.tracking(
        "doc_video.mp4", output_dir="jde_mot_result", visualization=True, draw_threshold=0.5, use_gpu=use_gpu
    )
    # or read from a image stream
    # with tracker.stream_mode(output_dir='image_stream_output', visualization=True, draw_threshold=0.5, use_gpu=True):
    #    tracker.predict([images])
    os.system("hub uninstall jde_darknet53")
