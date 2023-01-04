"""fairmot_dla34"""
import os
import paddlehub as hub
import paddle

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_fairmot_dla34_predict():
    """fairmot_dla34"""
    os.system("hub install fairmot_dla34")
    tracker = hub.Module(name="fairmot_dla34")
    # Read from a video file
    tracker.tracking("doc_video.mp4", output_dir="mot_result", visualization=True, draw_threshold=0.5, use_gpu=use_gpu)
    # or read from a image stream
    # with tracker.stream_mode(output_dir='image_stream_output', visualization=True, draw_threshold=0.5, use_gpu=True):
    #    tracker.predict([images])
    os.system("hub uninstall fairmot_dla34")
