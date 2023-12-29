# audio2caption -- Audio to caption converter

from paddlemix.appflow import Appflow
import paddle
paddle.seed(1024)
task = Appflow(app="audio2caption", models=[
               "conformer_u2pp_online_wenetspeech", "THUDM/chatglm-6b"])
audio_file = "./zh.wav"
prompt = (
    "描述这段话：{}."
)
result = task(audio=audio_file, prompt=prompt)['prompt']
print(result)
