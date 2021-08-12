"""
test paddle_serving_client.io
"""
import os
import pytest

from paddle_serving_client.io import inference_model_to_serving


class TestClientIO(object):
    """test client.io class"""

    def setup_class(self):
        """setup func"""
        self.dir = os.path.dirname(os.path.abspath(__file__))
        self.origin_model = f"{self.dir}/../data/ResNet50"

    @pytest.mark.api_clientIo_inferenceModelToServing_parameters
    def test_inference_model_to_serving(self):
        """test model convert"""
        feed_names, fetch_names = inference_model_to_serving(
            dirname=self.origin_model, model_filename="model", params_filename="params"
        )
        print("feed_names:", list(feed_names))
        print("fetch_names:", list(fetch_names))
        assert list(feed_names) == ["image"]
        assert list(fetch_names) == ["save_infer_model/scale_0.tmp_0"]

    @pytest.mark.api_clientIo_inferenceModelToServing_parameters
    def test_inference_model_to_serving_encrypt(self):
        """test encrypt convert"""
        feed_names, fetch_names = inference_model_to_serving(
            dirname=self.origin_model,
            model_filename="model",
            params_filename="params",
            serving_server="encrypt_server",
            serving_client="encrypt_client",
            encryption=True,
        )

        with open("./key", "rb") as f:
            key = f.read()

        print(key)
        print("feed_names:", list(feed_names))
        print("fetch_names:", list(fetch_names))
        assert key != b""
        assert list(feed_names) == ["image"]
        assert list(fetch_names) == ["save_infer_model/scale_0.tmp_0"]
