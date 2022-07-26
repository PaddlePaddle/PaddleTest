# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
model clip
"""
import paddle


def clip_model_extra_op(path_prefix=None, model_dir=None, output_model_path=None):
    """
    load inference model and clip extra op
    Args:
        path_prefix(str):input model path prefix
        model_dir(str):input model dir (for __model__ type)
        output_model_path(str):output model path prefix
    Returns:
        None
    """
    paddle.enable_static()
    if paddle.is_compiled_with_cuda():
        place = paddle.CUDAPlace(0)
    else:
        place = paddle.CPUPlace()
    exe = paddle.static.Executor(place)
    if model_dir:
        inference_program, feed_target_names, fetch_targets = paddle.fluid.io.load_inference_model(
            dirname=model_dir, executor=exe
        )
    else:
        inference_program, feed_target_names, fetch_targets = paddle.static.load_inference_model(path_prefix, exe)
    input_var_list = []
    for var in inference_program.list_vars():
        if var.name in feed_target_names:
            input_var_list.append(var)

    # 与模型原输入变量顺序对齐
    input_var_list_origin = []
    for name in feed_target_names:
        for var in input_var_list:
            if var.name == name:
                input_var_list_origin.append(var)
                break

    print("feed_target_names: {}".format(feed_target_names))
    print("input_var_list: {}".format(input_var_list))
    paddle.static.save_inference_model(
        path_prefix=output_model_path,
        feed_vars=input_var_list_origin,
        fetch_vars=fetch_targets,
        executor=exe,
        program=inference_program,
        clip_extra=True,
    )
    print("==== extra op attributes have been clipped from inference program ====")
