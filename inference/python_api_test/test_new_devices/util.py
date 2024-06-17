"""
util
"""
import os


def get_model_file(model_dir):
    """get_model_file"""
    model_file = None
    params_file = None
    for f in os.listdir(model_dir):
        if f.endswith(".pdmodel"):
            model_file = f
        elif f.endswith(".pdiparams"):
            params_file = f
    return model_file, params_file
