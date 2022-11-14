"""
benchmark base values
"""

trt_int8 = {
    "PPYOLOE": {
        "model_info": {
            "model_name": "PPYOLOE",
        },
        "jingdu": {
            "th": 0.05,
            "value": 0.5135882081820193,
            "unit": "mAP",
        },
        "xingneng": {
            "th": 0.05,
            "value": 380.8,
            "unit": "ms",
            "batch_size": 1,
        },
    },
    "PicoDet": {
        "model_info": {
            "model_name": "PicoDet",
        },
        "jingdu": {
            "th": 0.05,
            "value": 0.300434412153292,
            "unit": "mAP",
        },
        "xingneng": {
            "th": 0.05,
            "value": 26.9,
            "unit": "ms",
            "batch_size": 1,
        },
    },
    "PP-HumanSeg-Lite": {
        "model_info": {
            "model_name": "PP-HumanSeg-Lite",
        },
        "jingdu": {
            "th": 0.05,
            "value": 0.960031583569334,
            "unit": "mIoU",
        },
        "xingneng": {
            "th": 0.05,
            "value": 42.8,
            "unit": "ms",
            "batch_size": 1,
        },
    },
    "PP-Liteseg": {
        "model_info": {
            "model_name": "PP-Liteseg",
        },
        "jingdu": {
            "th": 0.05,
            "value": 0.7703976119566152,
            "unit": "mIoU",
        },
        "xingneng": {
            "th": 0.05,
            "value": 370.2,
            "unit": "ms",
            "batch_size": 1,
        },
    },
    "HRNet": {
        "model_info": {
            "model_name": "HRNet",
        },
        "jingdu": {
            "th": 0.05,
            "value": 0.7896978097502604,
            "unit": "mIoU",
        },
        "xingneng": {
            "th": 0.05,
            "value": 759.6,
            "unit": "ms",
            "batch_size": 1,
        },
    },
    "UNet": {
        "model_info": {
            "model_name": "UNet",
        },
        "jingdu": {
            "th": 0.05,
            "value": 0.649965905161135,
            "unit": "mIoU",
        },
        "xingneng": {
            "th": 0.05,
            "value": 2217.8,
            "unit": "ms",
            "batch_size": 1,
        },
    },
    "Deeplabv3-ResNet50": {
        "model_info": {
            "model_name": "Deeplabv3-ResNet50",
        },
        "jingdu": {
            "th": 0.05,
            "value": 0.7990287567610845,
            "unit": "mIoU",
        },
        "xingneng": {
            "th": 0.05,
            "value": 3181.1,
            "unit": "ms",
            "batch_size": 1,
        },
    },
}
