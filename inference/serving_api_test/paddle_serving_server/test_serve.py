"""
test paddle_serving_server.serve
"""
import argparse
import os
from multiprocessing import Process
import pytest
import numpy as np
import pynvml

from paddle_serving_server.serve import format_gpu_to_strlist
from paddle_serving_server.serve import is_gpu_mode
from paddle_serving_server.serve import start_gpu_card_model
from paddle_serving_client import Client
from paddle_serving_app.reader import Sequential, File2Image, Resize, CenterCrop
from paddle_serving_app.reader import RGB2BGR, Transpose, Div, Normalize

from util import *


class TestServe(object):
    """test serve module"""

    def setup_class(self):
        """setup func"""
        self.dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = f"{os.path.split(self.dir)[0]}/data/resnet_v2_50_imagenet_model"
        self.client_dir = f"{os.path.split(self.dir)[0]}/data/resnet_v2_50_imagenet_client"
        self.img_path = f"{self.dir}/../data/daisy.jpg"

    def predict(self, batch=False, batch_size=1):
        """predict by bRPC client"""
        client = Client()
        client.load_client_config(self.client_dir)
        client.connect(["127.0.0.1:9696"])

        seq = Sequential(
            [
                File2Image(),
                Resize(256),
                CenterCrop(224),
                RGB2BGR(),
                Transpose((2, 0, 1)),
                Div(255),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], True),
            ]
        )
        img = seq(self.img_path)

        if batch:
            img_batch = img[np.newaxis, :]
            img_batch = np.repeat(img_batch, repeats=batch_size, axis=0)
            fetch_map = client.predict(feed={"image": img_batch}, fetch=["score"], batch=batch)
        else:
            fetch_map = client.predict(feed={"image": img}, fetch=["score"], batch=False)

        result_class = np.argmax(fetch_map["score"], axis=1)
        result_prob = np.max(fetch_map["score"], axis=1)
        print("fetch_map:", fetch_map)
        print("class:", result_class)
        print("prob:", result_prob)
        return result_class.tolist(), result_prob.tolist()

    @pytest.mark.api_serverServe_formatGpuToStrList_parameters
    def test_format_gpu_to_strlist_with_int(self):
        """test format_gpu_to_strlist with int type"""
        assert format_gpu_to_strlist(2) == ["2"]

    @pytest.mark.api_serverServe_formatGpuToStrList_parameters
    def test_format_gpu_to_strlist_with_list(self):
        """test format_gpu_to_strlist with list type"""
        assert format_gpu_to_strlist(["3"]) == ["3"]
        assert format_gpu_to_strlist([""]) == ["-1"]
        assert format_gpu_to_strlist([]) == ["-1"]
        assert format_gpu_to_strlist([0, 1]) == ["0", "1"]
        assert format_gpu_to_strlist(["0,2", "1,3"]) == ["0,2", "1,3"]
        # with None
        assert format_gpu_to_strlist(None) == ["-1"]
        # with valid gpu id
        with pytest.raises(ValueError) as e:
            format_gpu_to_strlist(["1", "-2"])
        assert str(e.value) == "The input of gpuid error."
        with pytest.raises(ValueError) as e:
            format_gpu_to_strlist(["0,-1"])
        assert str(e.value) == "You can not use CPU and GPU in one model."

    @pytest.mark.api_serverServe_isGpuMode_parameters
    def test_is_gpu_mode(self):
        """test is_gpu_mode"""
        assert is_gpu_mode(["-1"]) is False
        assert is_gpu_mode(["0,1"]) is True

    @pytest.mark.api_serverServe_startGpuCardModel_exception
    def test_start_gpu_card_model_without_model(self):
        """test start_gpu_card_model in exception"""
        args = default_args()
        args.model = ""
        with pytest.raises(SystemExit) as e:
            start_gpu_card_model(gpu_mode=False, port=args.port, args=args)
        assert str(e.value) == "-1"

    @pytest.mark.api_serverServe_startGpuCardModel_parameters
    def test_start_gpu_card_model_with_single_model_cpu(self):
        """test start_gpu_card_model single model on cpu"""
        args = default_args()
        args.model = [self.model_dir]
        args.port = 9696

        p = Process(target=start_gpu_card_model, kwargs={"gpu_mode": False, "port": args.port, "args": args})
        p.start()
        os.system("sleep 5")

        assert count_process_num_on_port(9696) == 1
        assert check_gpu_memory(0) is False

        # batch = False
        brcp_class, brpc_prob = self.predict(batch=False)
        print(brcp_class, brpc_prob)
        assert brcp_class == [985]
        assert brpc_prob == [0.9341399073600769]

        # batch_size = 2
        brcp_class, brpc_prob = self.predict(batch=True, batch_size=2)
        print(brcp_class, brpc_prob)
        assert brcp_class == [985, 985]
        assert brpc_prob == [0.9341403245925903, 0.9341403245925903]

        kill_process(9696, 1)

    @pytest.mark.api_serverServe_startGpuCardModel_parameters
    def test_start_gpu_card_model_with_single_model_gpu(self):
        """test start_gpu_card_model single model on gpu"""
        args = default_args()
        args.model = [self.model_dir]
        args.port = 9696
        args.gpu_ids = ["0,1"]

        p = Process(target=start_gpu_card_model, kwargs={"gpu_mode": True, "port": args.port, "args": args})
        p.start()
        os.system("sleep 10")

        assert count_process_num_on_port(9696) == 1
        assert check_gpu_memory(0) is True
        assert check_gpu_memory(1) is True

        # batch = False
        brcp_class, brpc_prob = self.predict(batch=False)
        print(brcp_class, brpc_prob)
        assert brcp_class == [985]
        assert brpc_prob == [0.9341405034065247]

        # batch_size = 2
        brcp_class, brpc_prob = self.predict(batch=True, batch_size=2)
        print(brcp_class, brpc_prob)
        assert brcp_class == [985, 985]
        assert brpc_prob == [0.9341405034065247, 0.9341405034065247]

        kill_process(9696, 3)

    @pytest.mark.api_serverServe_startGpuCardModel_parameters
    def test_start_gpu_card_model_with_two_models_gpu(self):
        """test start_gpu_card_model two_models on gpu"""
        args = default_args()
        args.model = [self.model_dir, self.model_dir]
        args.port = 9696
        args.gpu_ids = ["0", "1"]

        p = Process(target=start_gpu_card_model, kwargs={"gpu_mode": True, "port": args.port, "args": args})
        p.start()
        os.system("sleep 10")

        assert count_process_num_on_port(9696) == 1
        assert check_gpu_memory(0) is True
        assert check_gpu_memory(1) is True

        kill_process(9696, 3)
