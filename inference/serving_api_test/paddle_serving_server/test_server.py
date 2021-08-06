"""
test paddle_serving_server.server
"""
import os
import time
from multiprocessing import Process
import pytest
import numpy as np

from paddle_serving_server.server import Server
import paddle_serving_server as serving
from paddle_serving_client import Client
from paddle_serving_app.reader import Sequential, File2Image, Resize, CenterCrop
from paddle_serving_app.reader import RGB2BGR, Transpose, Div, Normalize

from util import *


class TestServer(object):
    """test Server class (bRPC server)"""

    def setup_class(self):
        """setup func"""
        self.dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = f"{os.path.split(self.dir)[0]}/data/resnet_v2_50_imagenet_model"
        self.client_dir = f"{os.path.split(self.dir)[0]}/data/resnet_v2_50_imagenet_client"
        self.img_path = f"{self.dir}/../data/daisy.jpg"

    def setup_method(self):
        """setup func (init a standard server object)"""
        op_maker = serving.OpMaker()
        op_seq_maker = serving.OpSeqMaker()
        read_op = op_maker.create("general_reader")
        op_seq_maker.add_op(read_op)
        infer_op_name = "general_infer"
        general_infer_op = op_maker.create(infer_op_name)
        op_seq_maker.add_op(general_infer_op)
        general_response_op = op_maker.create("general_response")
        op_seq_maker.add_op(general_response_op)

        self.test_server = Server()
        self.test_server.set_op_sequence(op_seq_maker.get_op_sequence())
        self.test_server.load_model_config(self.model_dir)

    def teardown_method(self):
        """teardown func"""
        os.system("rm -rf workdir*")
        os.system("rm -rf PipelineServingLogs")
        os.system("rm -rf log")
        os.system("kill `ps -ef | grep serving | awk '{print $2}'` > /dev/null 2>&1")
        kill_process(9696)

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

    @pytest.mark.api_serverServer_loadModelConfig_parameters
    def test_load_model_config(self):
        """test load model config"""
        # check workflow_conf (already in test_dag.py)
        # check general_infer_0 op model_conf (feed_var and fetch_var)
        # feed_var
        feed_var = self.test_server.model_conf["general_infer_0"].feed_var
        assert feed_var[0].name == "image"
        assert feed_var[0].alias_name == "image"
        assert feed_var[0].is_lod_tensor is False
        assert feed_var[0].feed_type == 1
        assert feed_var[0].shape == [3, 224, 224]
        # fetch_var
        fetch_var = self.test_server.model_conf["general_infer_0"].fetch_var
        assert fetch_var[0].name == "softmax_0.tmp_0"
        assert fetch_var[0].alias_name == "score"
        assert fetch_var[0].is_lod_tensor is False
        assert fetch_var[0].fetch_type == 1
        assert fetch_var[0].shape == [1000]
        # check model_config_paths and server config filename
        assert self.test_server.model_config_paths["general_infer_0"] == self.model_dir
        assert self.test_server.general_model_config_fn == ["general_infer_0/general_model.prototxt"]
        assert self.test_server.model_toolkit_fn == ["general_infer_0/model_toolkit.prototxt"]
        assert self.test_server.subdirectory == ["general_infer_0"]

    @pytest.mark.api_serverServer_portIsAvailable_parameters
    def test_port_is_available_with_unused_port(self):
        """test port check"""
        assert self.test_server.port_is_available(12003) is True

    @pytest.mark.api_serverServer_portIsAvailable_parameters
    def test_port_is_available_with_used_port(self):
        """test port check in exception"""
        os.system("python -m SimpleHTTPServer 12005 &")
        time.sleep(2)
        assert self.test_server.port_is_available(12005) is False
        kill_process(12005)

    @pytest.mark.api_serverServer_checkAvx_parameters
    def test_check_avx(self):
        """test avx check"""
        assert self.test_server.check_avx() is True

    @pytest.mark.api_serverServer_getFetchList_parameters
    def test_get_fetch_list(self):
        """test get fetch list"""
        assert self.test_server.get_fetch_list() == ["score"]

    @pytest.mark.api_serverServer_prepareEngine_parameters
    def test_prepare_engine_with_async_mode(self):
        """test prepare server engine on async mode"""
        # 生成bRPC server配置信息(model_toolkit_conf)
        # check model_toolkit_conf
        self.test_server.set_op_num(4)
        self.test_server.set_op_max_batch(64)
        self.test_server.set_gpuid(["0,1"])
        self.test_server.set_gpu_multi_stream()
        self.test_server._prepare_engine(self.test_server.model_config_paths, "gpu", False)
        model_engine_0 = self.test_server.model_toolkit_conf[0].engines[0]

        assert model_engine_0.name == "general_infer_0"
        assert model_engine_0.type == "PADDLE_INFER"
        assert model_engine_0.reloadable_meta == f"{self.model_dir}/fluid_time_file"
        assert model_engine_0.reloadable_type == "timestamp_ne"
        assert model_engine_0.model_dir == self.model_dir
        assert model_engine_0.gpu_ids == [0, 1]
        assert model_engine_0.runtime_thread_num == 4
        assert model_engine_0.batch_infer_size == 64
        assert model_engine_0.enable_batch_align == 1
        assert model_engine_0.enable_memory_optimization is False
        assert model_engine_0.enable_ir_optimization is False
        assert model_engine_0.use_trt is False
        assert model_engine_0.use_lite is False
        assert model_engine_0.use_xpu is False
        assert model_engine_0.use_gpu is True
        assert model_engine_0.combined_model is False
        assert model_engine_0.gpu_multi_stream is True

    @pytest.mark.api_serverServer_prepareInferService_parameters
    def test_prepare_infer_service(self):
        """test prepare infer service conf"""
        # check infer_service_conf
        self.test_server._prepare_infer_service(9696)
        infer_service_conf = self.test_server.infer_service_conf

        assert infer_service_conf.port == 9696
        assert infer_service_conf.services[0].name == "GeneralModelService"
        assert infer_service_conf.services[0].workflows == ["workflow1"]

    @pytest.mark.api_serverServer_prepareResource_parameters
    def test_prepare_resource(self):
        """test prepare server resource"""
        # 生成模型feed_var,fetch_var配置文件(general_model.prototxt)，设置resource_conf属性
        # check resource_conf
        workdir = "workdir_9696"
        subdir = "general_infer_0"
        os.system("mkdir -p {}/{}".format(workdir, subdir))
        self.test_server._prepare_resource(workdir, None)
        resource_conf = self.test_server.resource_conf
        assert resource_conf.model_toolkit_path == ["workdir_9696"]
        assert resource_conf.model_toolkit_file == ["general_infer_0/model_toolkit.prototxt"]
        assert resource_conf.general_model_path == ["workdir_9696"]
        assert resource_conf.general_model_file == ["general_infer_0/general_model.prototxt"]

    @pytest.mark.api_serverServer_prepareServer_parameters
    def test_prepare_server(self):
        """test prepare server"""
        # 生成bRPC server各种配置文件
        self.test_server.prepare_server("workdir_9696", 9696, "gpu", False)
        assert os.path.isfile(f"{self.dir}/workdir_9696/general_infer_0/fluid_time_file") is True
        assert os.path.isfile(f"{self.dir}/workdir_9696/infer_service.prototxt") is True
        assert os.path.isfile(f"{self.dir}/workdir_9696/workflow.prototxt") is True
        assert os.path.isfile(f"{self.dir}/workdir_9696/resource.prototxt") is True
        assert os.path.isfile(f"{self.dir}/workdir_9696/general_infer_0/model_toolkit.prototxt") is True

    @pytest.mark.api_serverServer_runServer_parameters
    def test_run_server_with_cpu(self):
        """test run bRPC server on cpu"""
        self.test_server.prepare_server("workdir", 9696, "cpu")
        p = Process(target=self.test_server.run_server)
        p.start()
        os.system("sleep 5")

        assert check_gpu_memory(0) is False
        assert count_process_num_on_port(9696) == 1

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

    @pytest.mark.api_serverServer_runServer_parameters
    def test_run_server_with_gpu(self):
        """test run bRPC server on gpu"""
        self.test_server.set_gpuid("0,1")
        self.test_server.prepare_server("workdir_0", 9696, "gpu")
        p = Process(target=self.test_server.run_server)
        p.start()
        os.system("sleep 10")

        assert check_gpu_memory(0) is True
        assert check_gpu_memory(1) is True
        assert count_process_num_on_port(9696) == 1

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
