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

trt_fp16 = {
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
    "PP-HumanSeg-Lite": {
        "model_info": {
            "model_name": "PP-HumanSeg-Lite",
        },
        "jingdu": {
            "th": 0.05,
            "value": 0.9600320488643689,
            "unit": "mIoU",
        },
        "xingneng": {
            "th": 0.05,
            "value": 1.5,
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
            "value": 0.7703921401340169,
            "unit": "mIoU",
        },
        "xingneng": {
            "th": 0.05,
            "value": 10.7,
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
            "value": 0.7897009386638955,
            "unit": "mIoU",
        },
        "xingneng": {
            "th": 0.05,
            "value": 26.6,
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
            "value": 0.6499459695876132,
            "unit": "mIoU",
        },
        "xingneng": {
            "th": 0.05,
            "value": 48.9,
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
            "value": 0.7990384013842737,
            "unit": "mIoU",
        },
        "xingneng": {
            "th": 0.05,
            "value": 41.6,
            "unit": "ms",
            "batch_size": 1,
        },
    },
}

mkldnn_int8 = {
    "PPYOLOE": {
        "model_info": {
            "model_name": "PPYOLOE",
        },
        "jingdu": {
            "th": 0.05,
            "value": 0.5106316159315241,
            "unit": "mAP",
        },
        "xingneng": {
            "th": 0.05,
            "value": 345.4,
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
            "value": 0.2969912586510784,
            "unit": "mAP",
        },
        "xingneng": {
            "th": 0.05,
            "value": 22.3,
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
            "value": 0.9596980417424789,
            "unit": "mIoU",
        },
        "xingneng": {
            "th": 0.05,
            "value": 42.5,
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
            "value": 0.6688628078607249,
            "unit": "mIoU",
        },
        "xingneng": {
            "th": 0.05,
            "value": 300.0,
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
            "value": 0.7899464457999261,
            "unit": "mIoU",
        },
        "xingneng": {
            "th": 0.05,
            "value": 516.3,
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
            "value": 0.6434970135618086,
            "unit": "mIoU",
        },
        "xingneng": {
            "th": 0.05,
            "value": 1080.1,
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
            "value": 0.7900994083314681,
            "unit": "mIoU",
        },
        "xingneng": {
            "th": 0.05,
            "value": 843.7,
            "unit": "ms",
            "batch_size": 1,
        },
    },
}

mkldnn_fp32 = {
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
            "value": 403.6,
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
            "value": 25.6,
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
            "value": 43.1,
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
            "value": 356.3,
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
            "value": 718.1,
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
            "value": 2233.9,
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
            "value": 3188.2,
            "unit": "ms",
            "batch_size": 1,
        },
    },
}
