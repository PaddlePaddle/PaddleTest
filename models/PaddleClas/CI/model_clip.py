# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
clip model op attribute that marked as Extra
"""
import argparse
import os
import re

import paddle

def parse_args():
    """
    parse input args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir", type=str, default="./input_inference_model", help="input model path prefix")
    parser.add_argument(
        "--model_file", type=str, default=None, help="input model file_name")
    parser.add_argument(
        "--params_file", type=str, default=None, help="input params file_name")

    parser.add_argument(
        "--path_prefix", type=str, default=None, help="input model path prefix")
    parser.add_argument(
        "--output_model_path", type=str, default="./output_inference_model/inference", help="output model path prefix")
    return parser.parse_args()

def clip_model_extra_op(path_prefix=None, model_dir=None, output_model_path=None, **kwargs):
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

    if path_prefix and not model_dir:
        [inference_program, feed_target_names, fetch_targets] = (
            paddle.static.load_inference_model(path_prefix=path_prefix, executor=exe))
    elif model_dir and not path_prefix :
        model_file = kwargs.get("model_file", None)
        params_file = kwargs.get("params_file", None)

        if model_file and params_file:
            model_dir = model_file.split('/')[:-1]
            model_dir = "/".join(model_dir)
            [inference_program, feed_target_names, fetch_targets] = (
                paddle.fluid.io.load_inference_model(dirname=model_dir,
                executor=exe, model_filename=model_file, params_filename=params_file))
        else:
            [inference_program, feed_target_names, fetch_targets] = (
                paddle.fluid.io.load_inference_model(dirname=model_dir, 
                executor=exe))
    else:
        raise ValueError("==== please only set path_prefix or model_dir")

    input_var_list = []
    for var in inference_program.list_vars():
        if var.name in feed_target_names:
            input_var_list.append(var)

    print("feed_target_names: {}".format(feed_target_names))
    print("input_var_list: {}".format(input_var_list))
    paddle.static.save_inference_model(
        path_prefix=output_model_path,
        feed_vars=input_var_list,
        fetch_vars=fetch_targets,
        executor=exe,
        program=inference_program,
        clip_extra=True
    )
    print("==== extra op attributes have been clipped from inference program ====")
    print("==== op attribute cliped model has been saved to {0}".format(output_model_path))


if __name__ == "__main__":
    args = parse_args()
    model_style_2 = False
    path_prefix = args.path_prefix
    if path_prefix:
        model_root = path_prefix.split("/")
        if len(model_root) > 2:
            print("==== model_root[{}] length > 2".format(model_root))
            model_root.pop(-1)
            model_root = "/".join(model_root)
            for filename in os.listdir(model_root):
                root, ext = os.path.splitext(filename)
                if ext == '.pdmodel':
                    model_style_2 = True
        else:
            print("==== model_root[{}] length <= 2".format(model_root))
            model_style_2 = False

    if model_style_2:
        clip_model_extra_op(path_prefix=args.path_prefix, output_model_path=args.output_model_path)
    else:
        clip_model_extra_op(model_dir=args.model_dir,
                            model_file=args.model_file,
                            params_file=args.params_file,
                            output_model_path=args.output_model_path)

