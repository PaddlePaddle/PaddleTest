"""
test paddle_serving_server.web_service
"""
import os
from multiprocessing import Process
import base64
import numpy as np
import requests
import cv2
import pytest

from paddle_serving_server.web_service import WebService
import paddle_serving_server.serve
from paddle_serving_client import Client
from paddle_serving_app.reader import Sequential, File2Image, Resize, CenterCrop
from paddle_serving_app.reader import RGB2BGR, Transpose, Div, Normalize

from test_dag import TestOpSeqMaker
from util import *


class ResnetService(WebService):
    """Resnet web service class"""

    def init_imagenet_setting(self):
        """init web service settings"""
        self.seq = Sequential(
            [
                Resize(256),
                CenterCrop(224),
                RGB2BGR(),
                Transpose((2, 0, 1)),
                Div(255),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], True),
            ]
        )
        self.label_dict = {}
        label_idx = 0
        self.dir = os.path.dirname(os.path.abspath(__file__))
        self.label_dir = f"{self.dir}/../data/imagenet.label"
        with open(self.label_dir) as fin:
            for line in fin:
                self.label_dict[label_idx] = line.strip()
                label_idx += 1

    def preprocess(self, feed=[], fetch=[]):
        """web service preprocess"""
        # feed_batch最好直接封装为dict，local模式只支持dict类型
        # client.predict会将dict封装为len为1的list，dict的value为带有batch维的ndarray
        feed_batch = {}
        is_batch = True

        for i, _ in enumerate(feed):
            if "image" not in feed[i]:
                raise ValueError("feed data error!")
            data_str = base64.b64decode(feed[i]["image"].encode("utf8"))
            data = np.fromstring(data_str, np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            img = self.seq(img)
            if i == 0:
                img_batch = img[np.newaxis, :]
            else:
                img_batch = np.append(img_batch, img[np.newaxis, :], axis=0)
        feed_batch["image"] = img_batch
        return feed_batch, fetch, is_batch

    def postprocess(self, feed=[], fetch=[], fetch_map={}):
        """web service postprocess"""
        score_list = fetch_map["score"]
        result = {"label": [], "prob": []}
        for score in score_list:
            score = score.tolist()
            max_score = max(score)
            result["label"].append(self.label_dict[score.index(max_score)].strip().replace(",", ""))
            result["prob"].append(max_score)
        return result


class TestWebService(object):
    """test WebService class"""

    def setup_method(self):
        """setup func"""
        self.dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = f"{os.path.split(self.dir)[0]}/data/resnet_v2_50_imagenet_model"
        self.client_dir = f"{os.path.split(self.dir)[0]}/data/resnet_v2_50_imagenet_client"
        self.img_path = f"{self.dir}/../data/daisy.jpg"
        test_service = ResnetService("Resnet_service")
        test_service.load_model_config(self.model_dir)
        self.test_service = test_service

    def teardown(self):
        """teardown func"""
        os.system("rm -rf workdir*")
        os.system("rm -rf PipelineServingLogs")

    def predict_brpc(self, batch=False, batch_size=1):
        """predict by bRPC client"""
        client = Client()
        client.load_client_config(self.client_dir)
        client.connect(["127.0.0.1:12000"])

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

    def predict_http(self, port=9696, batch=False, batch_size=1):
        """predict by post HTTP request"""
        web_url = f"http://127.0.0.1:{port}/Resnet_service/prediction"
        with open(self.img_path, "rb") as file:
            image_data = file.read()
        image = cv2_to_base64(image_data)

        if batch:
            img_batch = []
            for i in range(batch_size):
                img_batch.append({"image": image})
        else:
            img_batch = [{"image": image}]

        data = {"feed": img_batch, "fetch": ["score"]}

        result = requests.post(url=web_url, json=data)
        print("http_result: ", result.json())
        return result.json()

    @pytest.mark.api_serverWebService_loadModelConfig_parameters
    def test_load_model_config(self):
        """test load model config"""
        # config_dir list
        assert self.test_service.server_config_dir_paths == [self.model_dir]
        # feed_vars
        feed_var = self.test_service.feed_vars["image"]
        assert feed_var.name == "image"
        assert feed_var.alias_name == "image"
        assert feed_var.is_lod_tensor is False
        assert feed_var.feed_type == 1
        assert feed_var.shape == [3, 224, 224]
        # fetch_vars
        fetch_var = self.test_service.fetch_vars["score"]
        assert fetch_var.name == "softmax_0.tmp_0"
        assert fetch_var.alias_name == "score"
        assert fetch_var.is_lod_tensor is False
        assert fetch_var.fetch_type == 1
        assert fetch_var.shape == [1000]
        # client config_path list
        assert self.test_service.client_config_path == [self.model_dir + "/serving_server_conf.prototxt"]

    @pytest.mark.api_serverWebService_prepareServer_parameters
    def test_prepare_server(self):
        """test prepare server"""
        self.test_service.prepare_server(workdir="workdir", port=9696, device="cpu")
        assert self.test_service.workdir == "workdir"
        assert self.test_service.port == 9696
        assert self.test_service.port_list == [12000]

    @pytest.mark.api_serverWebService_fefaultRpcService_parameters
    def test_default_rpc_service(self):
        """test init default rpc service"""
        self.test_service.prepare_server(workdir="workdir", port=9696, device="cpu")
        test_server = self.test_service.default_rpc_service(
            workdir="workdir", port=self.test_service.port_list[0], gpus=-1
        )
        # check bRPC server params
        assert test_server.port == 12000
        assert test_server.workdir == "workdir"
        assert test_server.device == "cpu"
        # check workflows list
        workflows = test_server.workflow_conf.workflows
        assert len(workflows) == 1
        TestOpSeqMaker.check_standard_workflow(workflows[0])

    @pytest.mark.api_serverWebService_createRpcConfig_parameters
    def test_create_rpc_config_with_cpu(self):
        """test create rpc service config on cpu"""
        self.test_service.prepare_server(workdir="workdir", port=9696, device="cpu")
        self.test_service.create_rpc_config()
        rpc_list = self.test_service.rpc_service_list
        brpc_server = rpc_list[0]

        assert brpc_server.device == "cpu"
        assert brpc_server.port == 12000
        assert brpc_server.workdir == "workdir"
        assert len(rpc_list) == 1
        assert isinstance(rpc_list[0], paddle_serving_server.server.Server)

    @pytest.mark.api_serverWebService_createRpcConfig_parameters
    def test_create_rpc_config_with_gpu(self):
        """test create rpc service config on gpu"""
        self.test_service.set_gpus("0,1")
        self.test_service.prepare_server(workdir="workdir", port=9696, device="gpu")
        self.test_service.create_rpc_config()
        rpc_list = self.test_service.rpc_service_list
        brpc_server = rpc_list[0]

        assert brpc_server.device == "gpu"
        assert brpc_server.gpuid == ["0,1"]
        assert brpc_server.port == 12000
        assert brpc_server.workdir == "workdir"
        assert len(rpc_list) == 1
        assert isinstance(rpc_list[0], paddle_serving_server.server.Server)

    @pytest.mark.api_serverWebService_setGpus_parameters
    def test_set_gpus(self):
        """test set gpu id"""
        self.test_service.set_gpus("1,2,3")
        assert self.test_service.gpus == ["1,2,3"]

    @pytest.mark.run(order=1)
    @pytest.mark.api_serverWebService_runWebService_parameters
    def test_run_web_service(self):
        """test run web service"""
        self.test_service.init_imagenet_setting()
        self.test_service.set_gpus("0,1")
        self.test_service.prepare_server(workdir="workdir", port=9393, device="gpu")
        self.test_service.run_rpc_service()
        p = Process(target=self.test_service.run_web_service)
        p.start()
        os.system("sleep 10")

        assert check_gpu_memory(0) is True
        assert check_gpu_memory(1) is True
        assert count_process_num_on_port(9393) == 1
        assert count_process_num_on_port(12000) == 1

        # batch = False
        http_result = self.predict_http(9393, batch=False)
        result_class = http_result["result"]["label"]
        result_prob = http_result["result"]["prob"]
        assert result_class == ["daisy"]
        assert result_prob == [0.9341405034065247]

        # batch_size = 2
        http_result = self.predict_http(9393, batch=True, batch_size=2)
        result_class = http_result["result"]["label"]
        result_prob = http_result["result"]["prob"]
        assert result_class == ["daisy", "daisy"]
        assert result_prob == [0.9341405034065247, 0.9341405034065247]

        # batch = False
        brcp_class, brpc_prob = self.predict_brpc(batch=False)
        print(brcp_class, brpc_prob)
        assert brcp_class == [985]
        assert brpc_prob == [0.9341405034065247]

        # batch_size = 2
        brcp_class, brpc_prob = self.predict_brpc(batch=True, batch_size=2)
        print(brcp_class, brpc_prob)
        assert brcp_class == [985, 985]
        assert brpc_prob == [0.9341405034065247, 0.9341405034065247]

        kill_process(9393)
        kill_process(12000, 3)

    @pytest.mark.api_serverWebService_runRpcService_parameters
    def test_run_rpc_service_with_gpu(self):
        """test only run rpc service on gpu"""
        self.test_service.set_gpus("0,1")
        self.test_service.prepare_server(workdir="workdir", port=9696, device="gpu")
        self.test_service.run_rpc_service()
        os.system("sleep 10")

        assert check_gpu_memory(0) is True
        assert check_gpu_memory(1) is True
        assert count_process_num_on_port(12000) == 1

        brcp_class, brpc_prob = self.predict_brpc(batch=False)
        print(brcp_class, brpc_prob)
        assert brcp_class == [985]
        assert brpc_prob == [0.9341405034065247]

        kill_process(12000, 3)

    @pytest.mark.api_serverWebService_runDebuggerService_parameters
    def test_run_debugger_service(self):
        """test local predict"""
        self.test_service.init_imagenet_setting()
        self.test_service.set_gpus("0")
        self.test_service.prepare_server(workdir="workdir", port=9696, device="gpu")
        self.test_service.run_debugger_service()
        p = Process(target=self.test_service.run_web_service)
        p.start()
        os.system("sleep 5")
        # TODO local模式直接使用paddle.inference进行推理，如何判断是否使用了GPU

        assert count_process_num_on_port(9696) == 1

        # batch = False
        http_result = self.predict_http(9696)
        result_class = http_result["result"]["label"]
        result_prob = http_result["result"]["prob"][0]
        assert result_class == ["daisy"]
        assert result_prob == 0.9341399073600769

        # batch_size = 2
        http_result = self.predict_http(9696, batch=True, batch_size=2)
        result_class = http_result["result"]["label"]
        result_prob = http_result["result"]["prob"]
        assert result_class == ["daisy", "daisy"]
        assert result_prob == [0.9341403245925903, 0.9341403245925903]

        kill_process(9696, 1)
