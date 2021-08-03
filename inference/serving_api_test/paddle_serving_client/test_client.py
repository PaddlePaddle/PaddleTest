"""
test paddle_serving_client.client
"""
import os
import sys
import subprocess
import numpy as np
import pytest

from paddle_serving_client import Client, MultiLangClient, SDKConfig
from paddle_serving_app.reader import Sequential, File2Image, Resize, CenterCrop
from paddle_serving_app.reader import RGB2BGR, Transpose, Div, Normalize

# pylint: disable=wrong-import-position
sys.path.append("../paddle_serving_server")
from util import default_args, kill_process, check_gpu_memory


class TestSDKConfig(object):
    """test SDKConfig"""

    @pytest.mark.api_clientClient_addServerVariant_parameters
    def test_add_server_variant(self):
        """test add server variant"""
        client = Client()
        predictor_sdk = SDKConfig()

        # check predictor_sdk
        client_id = id(client)
        predictor_sdk.add_server_variant("default_tag_{}".format(client_id), ["127.0.0.1:12003"], "100")
        assert predictor_sdk.tag_list == [f"default_tag_{client_id}"]
        assert predictor_sdk.cluster_list == [["127.0.0.1:12003"]]
        assert predictor_sdk.variant_weight_list == ["100"]

    @pytest.mark.api_clientClient_genDesc_parameters
    def test_gen_desc(self):
        """test sdk desc"""
        client = Client()
        predictor_sdk = SDKConfig()
        client_id = id(client)
        predictor_sdk.add_server_variant("default_tag_{}".format(client_id), ["127.0.0.1:12003"], "90")
        sdk_desc = predictor_sdk.gen_desc(300000)

        # check sdk_desc
        assert sdk_desc.default_variant_conf.connection_conf.rpc_timeout_ms == 300000
        assert sdk_desc.predictors[0].weighted_random_render_conf.variant_weight_list == "90"
        assert sdk_desc.predictors[0].variants[0].tag == f"default_tag_{client_id}"
        assert sdk_desc.predictors[0].variants[0].naming_conf.cluster == "list://127.0.0.1:12003"


class TestClient(object):
    """test brpc client"""

    def setup_class(self):
        """setup func"""
        self.dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = f"{os.path.split(self.dir)[0]}/paddle_serving_server/resnet_v2_50_imagenet_model"
        self.client_dir = f"{os.path.split(self.dir)[0]}/paddle_serving_server/resnet_v2_50_imagenet_client"
        with open(f"{self.model_dir}/../key", "rb") as f:
            self.key = f.read()

    def setup_method(self):
        """setup func"""
        self.dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = f"{os.path.split(self.dir)[0]}/paddle_serving_server/resnet_v2_50_imagenet_model"
        self.client_dir = f"{os.path.split(self.dir)[0]}/paddle_serving_server/resnet_v2_50_imagenet_client"

    @pytest.mark.api_clientClient_loadClientConfig_parameters
    def test_load_client_config(self):
        """check client config"""
        client = Client()
        client.load_client_config(self.client_dir)
        # check feed_names_ feed_names_to_idx and so on (feed_vars)
        assert client.feed_names_ == ["image"]
        assert client.feed_names_to_idx_ == {"image": 0}
        assert client.feed_types_ == {"image": 1}
        assert client.feed_shapes_ == {"image": [3, 224, 224]}
        assert client.feed_tensor_len == {"image": 3 * 224 * 224}
        # check fetch_vars(多模型串联时由最后一个模型决定)
        assert client.fetch_names_ == ["score"]
        assert client.fetch_names_to_idx_ == {"score": 0}
        assert client.fetch_names_to_type_ == {"score": 1}
        # TODO client.client_handle_(PredictorClient对象)

    @pytest.mark.api_clientClient_useKey_parameters
    def test_use_key(self):
        """test encrypt key"""
        client = Client()
        client.load_client_config(self.client_dir)
        client.use_key(f"{self.model_dir}/../key")

        # check key
        assert client.key == self.key

    @pytest.mark.api_clientClient_getServingPort_parameters
    def test_get_serving_port(self):
        """test brpc server port"""
        # start encrypt server
        p = subprocess.Popen(
            f"python3.6 -m paddle_serving_server.serve --model "
            f"{os.path.split(self.dir)[0]}/paddle_serving_server/encrypt_server --port 9300 --use_encryption_model",
            shell=True,
        )
        os.system("sleep 3")

        client = Client()
        client.load_client_config(self.client_dir)
        client.use_key("./key")
        endpoints = client.get_serving_port(["127.0.0.1:9300"])
        assert endpoints == ["127.0.0.1:12000"]

        os.system("sleep 3")
        kill_process(12000)
        kill_process(9300)

    @pytest.mark.api_clientClient_shapeCheck_parameters
    def test_shape_check(self):
        """test tensor shape check"""
        # shape_check only check list type
        client = Client()
        client.load_client_config(self.client_dir)

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
        image_file = f"{self.model_dir}/../daisy.jpg"
        img = seq(image_file)
        img = img.flatten().tolist()
        feed = {"image": img}

        client.shape_check(feed, "image")

    @pytest.mark.api_clientClient_loadClientConfig_exception
    def test_shape_check_with_wrong_shape(self):
        """test in exception"""
        client = Client()
        client.load_client_config(self.client_dir)

        feed = {"image": [1.2, 1.3, 1.4]}

        with pytest.raises(ValueError) as e:
            client.shape_check(feed, "image")
        assert str(e.value) == f"The shape of feed tensor image not match."

    @pytest.mark.api_clientClient_predict_parameters
    def test_predict(self):
        """test predict with GPU server"""
        p = subprocess.Popen(
            f"python3.6 -m paddle_serving_server.serve --model {self.model_dir} --port 9697 --gpu_ids 0,1", shell=True
        )
        os.system("sleep 10")

        assert check_gpu_memory(0) is True
        assert check_gpu_memory(1) is True

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
        image_file = f"{self.model_dir}/../daisy.jpg"
        img = seq(image_file)
        feed = {"image": img}

        client = Client()
        client.load_client_config(self.client_dir)
        client.connect(["127.0.0.1:9697"])

        # batch False
        fetch_map = client.predict(feed=feed, fetch=["score"], batch=False)
        result_class = np.argmax(fetch_map["score"], axis=1).tolist()
        result_prob = np.max(fetch_map["score"], axis=1).tolist()
        print("result_class:", result_class)
        print("result_prob:", result_prob)
        assert result_class == [985]
        assert result_prob == [0.9341405034065247]

        # batch_size = 2
        img_batch = img[np.newaxis, :]
        img_batch = np.repeat(img_batch, repeats=2, axis=0)
        fetch_map = client.predict(feed={"image": img_batch}, fetch=["score"], batch=True)
        result_class = np.argmax(fetch_map["score"], axis=1).tolist()
        result_prob = np.max(fetch_map["score"], axis=1).tolist()
        print("result_class:", result_class)
        print("result_prob:", result_prob)
        assert result_class == [985, 985]
        assert result_prob == [0.9341405034065247, 0.9341405034065247]

        kill_process(9697, 3)


class TestMultiLangClient(object):
    """test grpc client"""

    def setup_method(self):
        """setup func"""
        self.dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = f"{os.path.split(self.dir)[0]}/paddle_serving_server/resnet_v2_50_imagenet_model"
        self.client_dir = f"{os.path.split(self.dir)[0]}/paddle_serving_server/resnet_v2_50_imagenet_client"

    def start_grpc_server_with_bsf(self):
        """start an async grpc server"""
        p = subprocess.Popen(
            f"python3.6 -m paddle_serving_server.serve --model {self.model_dir} "
            f"--port 9696 --gpu_ids 0,1 --thread 10 --op_num 2 --use_multilang",
            shell=True,
        )
        os.system("sleep 10")

    @pytest.mark.api_clientClient_connect_parameters
    def test_connect(self):
        """test client connect"""
        self.start_grpc_server_with_bsf()

        assert check_gpu_memory(0) is True
        assert check_gpu_memory(1) is True

        client = MultiLangClient()
        client.connect(["127.0.0.1:9696"])
        assert client.feed_names_ == ["image"]
        assert client.feed_types_ == {"image": 1}
        assert client.feed_shapes_ == {"image": [3, 224, 224]}
        assert client.fetch_names_ == ["score"]
        assert client.fetch_types_ == {"score": 1}

        kill_process(9696)
        kill_process(12000, 3)

    @pytest.mark.api_clientClient_predict_parameters
    def test_predict(self):
        """test predict"""
        self.start_grpc_server_with_bsf()

        assert check_gpu_memory(0) is True
        assert check_gpu_memory(1) is True

        client = MultiLangClient()
        client.connect(["127.0.0.1:9696"])

        # prepared image data
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
        image_file = f"{self.model_dir}/../daisy.jpg"
        img = seq(image_file)

        # sync
        for i in range(1, 3):
            feed = {"image": img}
            fetch_map = client.predict(feed=feed, fetch=["score"], batch=False)
            print(fetch_map)
            result_class = np.argmax(fetch_map["score"], axis=1).tolist()
            result_prob = np.max(fetch_map["score"], axis=1).tolist()
            print(f"sync_turn{i}_result_class:", result_class)
            print(f"sync_turn{i}_result_prob:", result_prob)
            assert result_class == [985]
            assert result_prob == [0.9341405034065247]

        # batch_size = 2
        img_batch = img[np.newaxis, :]
        img_batch = np.repeat(img_batch, repeats=2, axis=0)
        for i in range(1, 3):
            fetch_map = client.predict(feed={"image": img_batch}, fetch=["score"], batch=True)
            result_class = np.argmax(fetch_map["score"], axis=1).tolist()
            result_prob = np.max(fetch_map["score"], axis=1).tolist()
            print(f"batch_turn{i}_result_class:", result_class)
            print(f"batch_turn{i}_result_prob:", result_prob)
            assert result_class == [985, 985]
            assert result_prob == [0.9341405034065247, 0.9341405034065247]

        # async
        for i in range(1, 3):
            feed = {"image": img}
            future = client.predict(feed=feed, fetch=["score"], batch=False, asyn=True)
            fetch_map = future.result()
            result_class = np.argmax(fetch_map["score"], axis=1).tolist()
            result_prob = np.max(fetch_map["score"], axis=1).tolist()
            print(f"asyn_turn{i}_result_class:", result_class)
            print(f"asyn_turn{i}_result_prob:", result_prob)
            assert result_class == [985]
            assert result_prob == [0.9341405034065247]

        kill_process(9696)
        kill_process(12000, 3)
