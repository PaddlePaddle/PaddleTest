# audio_chat
from paddlemix.appflow import Appflow
import paddle
paddle.seed(1024)
task = Appflow(app="audio_chat", models=[
               "conformer_u2pp_online_wenetspeech", "THUDM/chatglm-6b", "speech"])
audio_file = "./zh.wav"
prompt = (
    "描述这段话：{}."
)
output_path = "tmp.wav"
result = task(audio=audio_file, prompt=prompt, output=output_path)
