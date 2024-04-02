"""
encrypt model
"""
from paddle_serving_client.io import inference_model_to_serving


def serving_encryption():
    """encrypt model to serving_model"""
    inference_model_to_serving(
        dirname="./resnet_v2_50_imagenet_model",
        params_filename=None,
        serving_server="encrypt_server",
        serving_client="encrypt_client",
        encryption=True,
    )


if __name__ == "__main__":
    serving_encryption()
