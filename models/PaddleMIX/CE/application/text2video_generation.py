from paddlemix.appflow import Appflow
from PIL import Image


prompt = "An astronaut riding a horse."

app = Appflow(app='text_to_video_generation',models=['damo-vilab/text-to-video-ms-1.7b'])
video_frames = app(prompt=prompt,num_inference_steps=25)['result']

imageio.mimsave("text_to_video_generation-synth-result-astronaut_riding_a_horse.gif", video_frames,duration=8)