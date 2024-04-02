"""stable_diffusion"""
import os
import paddlehub as hub
import paddle
import cv2

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_stable_diffusion_predict():
    """
    stable_diffusion
    """
    os.system("hub install stable_diffusion")
    module = hub.Module(name="stable_diffusion")
    text_prompts = [
        "in the morning light,Overlooking TOKYO city by greg rutkowski and thomas kinkade,Trending on artstation."
    ]
    # 生成图像, 默认会在stable_diffusion_out目录保存图像
    # 返回的da是一个DocumentArray对象，保存了所有的结果，包括最终结果和迭代过程的中间结果
    # 可以通过操作DocumentArray对象对生成的图像做后处理，保存或者分析
    # 您可以设置batch_size一次生成多张
    da = module.generate_image(text_prompts=text_prompts, batch_size=3, output_dir="./stable_diffusion_out/")
    # 展示所有的中间结果
    da[0].chunks[-1].chunks.plot_image_sprites(skip_empty=True, show_index=True, keep_aspect_ratio=True)
    # 将整个生成过程保存为一个动态图gif
    da[0].chunks[-1].chunks.save_gif("stable_diffusion_out-merged-result.gif")
    # da索引的是prompt, da[0].chunks索引的是该prompt下生成的第一张图，在batch_size不为1时能同时生成多张图
    # 您也可以按照上述操作显示单张图，如第0张的生成过程
    da[0].chunks[0].chunks.plot_image_sprites(skip_empty=True, show_index=True, keep_aspect_ratio=True)
    da[0].chunks[0].chunks.save_gif("stable_diffusion_out-image-0-result.gif")
    os.system("hub uninstall stable_diffusion")
