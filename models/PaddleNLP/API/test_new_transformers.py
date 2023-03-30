import importlib
import paddle
import pytest
from paddlenlp.transformers.auto.modeling import *
from paddlenlp.transformers import *
from paddlenlp.transformers.auto.modeling import get_name_mapping
from paddlenlp.transformers.utils import (
    find_transformer_model_class_by_name,
    import_module,
    find_transformer_model_type,
)
from paddlenlp.transformers.blip.modeling import BLIP_PRETRAINED_MODEL_ARCHIVE_LIST
from paddlenlp.transformers.pegasus.tokenizer import PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
from paddlenlp.transformers.chineseclip.modeling import CHINESE_CLIP_PRETRAINED_MODEL_ARCHIVE_LIST
from paddlenlp.transformers.ernie_vil.modeling import ERNIE_VIL_PRETRAINED_MODEL_ARCHIVE_LIST
from paddlenlp.utils.env import PPNLP_HOME
from paddlenlp.utils.log import logger

""" resolve issues https://github.com/PaddlePaddle/PaddleNLP/issues/5158 """


def get_new_transformers_models():
    """get new transformers models by auto.get_name_mapping()

    Returns:
            PretrainedModel: An instance of `AutoModelClass`.
    Example:
            "MT5Model",
            "MT5PretrainedModel",
            "MT5ForConditionalGeneration",
            "MT5EncoderModel"]
    """

    model_class_list = []
    for key, values in enumerate(get_name_mapping()):
        if values.endswith("Model"):
            return model_class_list.append(values)
    return model_class_list


@pytest.mark.parametrize("model_name", get_new_transformers_models())
def test_pretrained_resource_files(model_name):
    """get pretrained resource files by auto.get_init_configurations()

    Example:
            MT5_PRETRAINED_MODEL_ARCHIVE_LIST = [
            "google/mt5-small",
            "google/mt5-base",
            "google/mt5-large",
            "google/mt5-xl",
            "google/mt5-xxl"]
    """
    pretrained_resource_list = []
    NO_pretrained_resource_files_map_model_list = []
    ModelClasse = find_transformer_model_class_by_name(model_name)
    try:
        # 1. get PretrainedModel cls.from pretrained_resource_files_map
        resource_names = list(ModelClasse.pretrained_resource_files_map["model_state"].keys())
        pretrained_resource_list.extend(resource_names)

        # configurations = ModelClasse.pretrained_init_configuration
        # model_type = find_transformer_model_type(ModelClasse)

    except Exception:
        try:
            # 2. get PretrainedModel from cls.PRETRAINED_MODEL_ARCHIVE_LIST
            model_archive_name = model_name.rstrip("Model") + "_PRETRAINED_MODEL_ARCHIVE_LIST"
            lower_values = model_name.rstrip("Model").lower()
            module = importlib.import_module(f"paddlenlp.transformers.{lower_values}.modeling")

            model_archive_name_attr = getattr(module, model_archive_name)
            pretrained_resource_list.extend(model_archive_name_attr)

        except Exception:
            NO_pretrained_resource_files_map_model_list.append(model_name)
            logger.warning(f"{model_name} Not definition PRETRAINED_RESOURCE_FILES_MAP ")

    # 3. get PretrainedModel from others
    pretrained_resource_list.extend(BLIP_PRETRAINED_MODEL_ARCHIVE_LIST)
    pretrained_resource_list.extend(PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES.keys())
    pretrained_resource_list.extend(CHINESE_CLIP_PRETRAINED_MODEL_ARCHIVE_LIST)
    pretrained_resource_list.extend(ERNIE_VIL_PRETRAINED_MODEL_ARCHIVE_LIST)

    return pretrained_resource_list


def test_pretrained_resource_files_map():
    """get pretrained model list from cls.pretrained_resource_files_map"""
    pass


def test_PRETRAINED_MODEL_ARCHIVE_LIST():
    """get pretrained model list from cls_PRETRAINED_MODEL_ARCHIVE_LIST"""
    pass


def test_pretrained_init_configuration():
    """get pretrained model list from cls.pretrained_init_configuration"""
    pass


@pytest.mark.parametrize("resource_model_name", test_pretrained_resource_files())
def test_Pretrained_Models_and_Tokenizer(resource_model_name):
    """
    test download save/from_pretrained Models & Tokenizer
    """
    # 1. load from bos url
    model_pretrained = AutoModel.from_pretrained(resource_model_name)
    tokenizer = AutoTokenizer.from_pretrained(resource_model_name)

    # 2. load from cache
    model_pretrained.save_pretrained(PPNLP_HOME + resource_model_name)
    tokenizer.save_pretrained(PPNLP_HOME + resource_model_name)

    # 3. load from local
    load_from_model_local = AutoModel.from_pretrained(PPNLP_HOME + resource_model_name)
    load_from_tokenizer_local = AutoTokenizer.from_pretrained(PPNLP_HOME + resource_model_name)

    # 4. paddle load
    model_state = paddle.load(PPNLP_HOME + resource_model_name + "/model_state.pdparams")
    model_pretrained.set_state_dict(model_state)
