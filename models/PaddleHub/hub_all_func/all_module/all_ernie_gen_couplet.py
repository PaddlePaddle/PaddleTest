"""ernie_gen_couplet"""
import os
import paddlehub as hub
import paddle

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_ernie_gen_couplet_predict():
    """ernie_gen_couplet"""
    os.system("hub install ernie_gen_couplet")
    module = hub.Module(name="ernie_gen_couplet")

    test_texts = ["人增福寿年增岁", "风吹云乱天垂泪"]
    results = module.generate(texts=test_texts, use_gpu=use_gpu, beam_width=5)
    for result in results:
        print(result)

    # ['春满乾坤喜满门', '竹报平安梅报春', '春满乾坤福满门', '春满乾坤酒满樽', '春满乾坤喜满家']
    # ['雨打花残地痛心', '雨打花残地皱眉', '雨打花残地动容', '雨打霜欺地动容', '雨打花残地洒愁']
    os.system("hub uninstall ernie_gen_couplet")
