#!/bin/bash

mkdir cat_toy_images
export http_proxy=${proxy}
export https_proxy=${proxy}
wget https://huggingface.co/sd-dreambooth-library/cat-toy/resolve/main/concept_images/0.jpeg -P ./cat_toy_images/
wget https://huggingface.co/sd-dreambooth-library/cat-toy/resolve/main/concept_images/1.jpeg -P ./cat_toy_images/
wget https://huggingface.co/sd-dreambooth-library/cat-toy/resolve/main/concept_images/2.jpeg -P ./cat_toy_images/
wget https://huggingface.co/sd-dreambooth-library/cat-toy/resolve/main/concept_images/3.jpeg -P ./cat_toy_images/
unset http_proxy
unset https_proxy