"""
download dataset
"""

import os
import zipfile
import wget

datasets = "https://dataset.bj.bcebos.com/PaddleScience/cylinder2D_continuous/datasets.zip"

wget.download(datasets)

with zipfile.ZipFile("datasets.zip", "r") as zip_ref:
    zip_ref.extractall("./")
