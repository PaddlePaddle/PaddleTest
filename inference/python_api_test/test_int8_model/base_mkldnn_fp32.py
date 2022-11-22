"""
mkldnn_fp32 base values
"""

mkldnn_fp32 = {
    "PPYOLOE": {
        "model_name": "PPYOLOE",
        "jingdu": {
            "value": 0.5135882081820193,
            "unit": "mAP",
            "th": 0.05,
        },
        "xingneng": {
            "value": 273.6,
            "unit": "ms",
            "batch_size": 1,
            "th": 0.05,
        },
    },
    "PicoDet": {
        "model_name": "PicoDet",
        "jingdu": {
            "value": 0.300434412153292,
            "unit": "mAP",
            "th": 0.05,
        },
        "xingneng": {
            "value": 17.4,
            "unit": "ms",
            "batch_size": 1,
            "th": 0.05,
        },
    },
    "YOLOv5s": {
        "model_name": "YOLOv5s",
        "jingdu": {
            "value": 0.37574151469621125,
            "unit": "mAP",
            "th": 0.05,
        },
        "xingneng": {
            "value": 40.5,
            "unit": "ms",
            "batch_size": 1,
            "th": 0.05,
        },
    },
    "YOLOv6s": {
        "model_name": "YOLOv6s",
        "jingdu": {
            "value": 0.42524875891435443,
            "unit": "mAP",
            "th": 0.05,
        },
        "xingneng": {
            "value": 58.7,
            "unit": "ms",
            "batch_size": 1,
            "th": 0.05,
        },
    },
    "YOLOv7": {
        "model_name": "YOLOv7",
        "jingdu": {
            "value": 0.5106915816882776,
            "unit": "mAP",
            "th": 0.05,
        },
        "xingneng": {
            "value": 136.2,
            "unit": "ms",
            "batch_size": 1,
            "th": 0.05,
        },
    },
    "ResNet_vd": {
        "model_name": "ResNet_vd",
        "jingdu": {
            "value": 0.79046,
            "unit": "acc",
            "th": 0.05,
        },
        "xingneng": {
            "value": 13.2,
            "unit": "ms",
            "batch_size": 1,
            "th": 0.05,
        },
    },
    "MobileNetV3_large": {
        "model_name": "MobileNetV3_large",
        "jingdu": {
            "value": 0.74958,
            "unit": "acc",
            "th": 0.05,
        },
        "xingneng": {
            "value": 5.2,
            "unit": "ms",
            "batch_size": 1,
            "th": 0.05,
        },
    },
    "PPLCNetV2": {
        "model_name": "PPLCNetV2",
        "jingdu": {
            "value": 0.76868,
            "unit": "acc",
            "th": 0.05,
        },
        "xingneng": {
            "value": 5.1,
            "unit": "ms",
            "batch_size": 1,
            "th": 0.05,
        },
    },
    "PPHGNet_tiny": {
        "model_name": "PPHGNet_tiny",
        "jingdu": {
            "value": 0.79594,
            "unit": "acc",
            "th": 0.05,
        },
        "xingneng": {
            "value": 12.4,
            "unit": "ms",
            "batch_size": 1,
            "th": 0.05,
        },
    },
    "EfficientNetB0": {
        "model_name": "EfficientNetB0",
        "jingdu": {
            "value": 0.77026,
            "unit": "acc",
            "th": 0.05,
        },
        "xingneng": {
            "value": 9.8,
            "unit": "ms",
            "batch_size": 1,
            "th": 0.05,
        },
    },
    "PP-HumanSeg-Lite": {
        "model_name": "PP-HumanSeg-Lite",
        "jingdu": {
            "value": 0.960031583569334,
            "unit": "mIoU",
            "th": 0.05,
        },
        "xingneng": {
            "value": 41.5,
            "unit": "ms",
            "batch_size": 1,
            "th": 0.05,
        },
    },
    "PP-Liteseg": {
        "model_name": "PP-Liteseg",
        "jingdu": {
            "value": 0.7703976119566152,
            "unit": "mIoU",
            "th": 0.05,
        },
        "xingneng": {
            "value": 419.6,
            "unit": "ms",
            "batch_size": 1,
            "th": 0.05,
        },
    },
    "HRNet": {
        "model_name": "HRNet",
        "jingdu": {
            "value": 0.7896978097502604,
            "unit": "mIoU",
            "th": 0.05,
        },
        "xingneng": {
            "value": 737.4,
            "unit": "ms",
            "batch_size": 1,
            "th": 0.05,
        },
    },
    "UNet": {
        "model_name": "UNet",
        "jingdu": {
            "value": 0.649965905161135,
            "unit": "mIoU",
            "th": 0.05,
        },
        "xingneng": {
            "value": 2234.3,
            "unit": "ms",
            "batch_size": 1,
            "th": 0.05,
        },
    },
    "Deeplabv3-ResNet50": {
        "model_name": "Deeplabv3-ResNet50",
        "jingdu": {
            "value": 0.7990287567610845,
            "unit": "mIoU",
            "th": 0.05,
        },
        "xingneng": {
            "value": 2806.4,
            "unit": "ms",
            "batch_size": 1,
            "th": 0.05,
        },
    },
    "ERNIE_3.0-Medium": {
        "model_name": "ERNIE_3.0-Medium",
        "jingdu": {
            "value": 0.7534754402224282,
            "unit": "acc",
            "th": 0.05,
        },
        "xingneng": {
            "value": 187.05,
            "unit": "ms",
            "batch_size": 32,
            "th": 0.05,
        },
    },
    "PP-MiniLM": {
        "model_name": "PP-MiniLM",
        "jingdu": {
            "value": 0.7402687673772012,
            "unit": "acc",
            "th": 0.05,
        },
        "xingneng": {
            "value": 180.81,
            "unit": "ms",
            "batch_size": 32,
            "th": 0.05,
        },
    },
    "BERT_Base": {
        "model_name": "BERT_Base",
        "jingdu": {
            "value": 0.6006530974238766,
            "unit": "acc",
            "th": 0.05,
        },
        "xingneng": {
            "value": 52.91,
            "unit": "ms",
            "batch_size": 1,
            "th": 0.05,
        },
    },
}
