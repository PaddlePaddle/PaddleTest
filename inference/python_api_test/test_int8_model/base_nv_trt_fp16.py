"""
nv_trt_fp16 base
"""

nv_trt_fp16 = {
    "PPYOLOE_PLUS": {
        "model_name": "PPYOLOE_PLUS",
        "batch_size": 1,
        "jingdu": {
            "value": 0.5600042618262268,
            "unit": "mAP",
            "th": 0.01
        },
        "xingneng": {
            "value": 3.2,
            "unit": "ms",
            "th": 0.05
        },
        "cpu_mem": {
            "value": 2433.3140599999997,
            "unit": "MB",
            "th": 0.05
        },
        "gpu_mem": {
            "value": 645.0,
            "unit": "MB",
            "th": 0.05
        }
    },
    "PicoDet": {
        "model_name": "PicoDet",
        "batch_size": 1,
        "jingdu": {
            "value": 0.393377893664232,
            "unit": "mAP",
            "th": 0.01
        },
        "xingneng": {
            "value": 1.6599999999999997,
            "unit": "ms",
            "th": 0.05
        },
        "cpu_mem": {
            "value": 2424.47658,
            "unit": "MB",
            "th": 0.05
        },
        "gpu_mem": {
            "value": 611.0,
            "unit": "MB",
            "th": 0.05
        }
    },
    "YOLOv5s": {
        "model_name": "YOLOv5s",
        "batch_size": 1,
        "jingdu": {
            "value": 0.475003852545204,
            "unit": "mAP",
            "th": 0.01
        },
        "xingneng": {
            "value": 3.88,
            "unit": "ms",
            "th": 0.05
        },
        "cpu_mem": {
            "value": 1334.86012,
            "unit": "MB",
            "th": 0.05
        },
        "gpu_mem": {
            "value": 299.0,
            "unit": "MB",
            "th": 0.05
        }
    },
    "YOLOv6s": {
        "model_name": "YOLOv6s",
        "batch_size": 1,
        "jingdu": {
            "value": 0.6171771559112594,
            "unit": "mAP",
            "th": 0.01
        },
        "xingneng": {
            "value": 3.2,
            "unit": "ms",
            "th": 0.05
        },
        "cpu_mem": {
            "value": 1318.9125,
            "unit": "MB",
            "th": 0.05
        },
        "gpu_mem": {
            "value": 307.0,
            "unit": "MB",
            "th": 0.05
        }
    },
    "YOLOv7": {
        "model_name": "YOLOv7",
        "batch_size": 1,
        "jingdu": {
            "value": 0.5972319038861243,
            "unit": "mAP",
            "th": 0.01
        },
        "xingneng": {
            "value": 8.48,
            "unit": "ms",
            "th": 0.05
        },
        "cpu_mem": {
            "value": 1342.5508000000002,
            "unit": "MB",
            "th": 0.05
        },
        "gpu_mem": {
            "value": 403.0,
            "unit": "MB",
            "th": 0.05
        }
    },
    "ResNet_vd": {
        "model_name": "ResNet_vd",
        "batch_size": 1,
        "jingdu": {
            "value": 0.7950049950049951,
            "unit": "acc",
            "th": 0.01
        },
        "xingneng": {
            "value": 1.1,
            "unit": "ms",
            "th": 0.05
        },
        "gpu_mem": {
            "value": 1339.4,
            "unit": "MB",
            "th": 0.05
        },
        "cpu_mem": {
            "value": 4842.69376,
            "unit": "MB",
            "th": 0.05
        }
    },
    "MobileNetV3_large": {
        "model_name": "MobileNetV3_large",
        "batch_size": 1,
        "jingdu": {
            "value": 0.7402597402597403,
            "unit": "acc",
            "th": 0.01
        },
        "xingneng": {
            "value": 0.7,
            "unit": "ms",
            "th": 0.05
        },
        "gpu_mem": {
            "value": 1311.4,
            "unit": "MB",
            "th": 0.05
        },
        "cpu_mem": {
            "value": 4634.945319999999,
            "unit": "MB",
            "th": 0.05
        }
    },
    "PPLCNetV2": {
        "model_name": "PPLCNetV2",
        "batch_size": 1,
        "jingdu": {
            "value": 0.7702297702297702,
            "unit": "acc",
            "th": 0.01
        },
        "xingneng": {
            "value": 0.54,
            "unit": "ms",
            "th": 0.05
        },
        "gpu_mem": {
            "value": 1311.4,
            "unit": "MB",
            "th": 0.05
        },
        "cpu_mem": {
            "value": 4619.91562,
            "unit": "MB",
            "th": 0.05
        }
    },
    "PPHGNet_tiny": {
        "model_name": "PPHGNet_tiny",
        "batch_size": 1,
        "jingdu": {
            "value": 0.8095904095904096,
            "unit": "acc",
            "th": 0.01
        },
        "xingneng": {
            "value": 1.2,
            "unit": "ms",
            "th": 0.05
        },
        "gpu_mem": {
            "value": 1328.2,
            "unit": "MB",
            "th": 0.05
        },
        "cpu_mem": {
            "value": 4705.87656,
            "unit": "MB",
            "th": 0.05
        }
    },
    "EfficientNetB0": {
        "model_name": "EfficientNetB0",
        "batch_size": 1,
        "jingdu": {
            "value": 0.7632367632367633,
            "unit": "acc",
            "th": 0.01
        },
        "xingneng": {
            "value": 1.0,
            "unit": "ms",
            "th": 0.05
        },
        "gpu_mem": {
            "value": 1313.0,
            "unit": "MB",
            "th": 0.05
        },
        "cpu_mem": {
            "value": 4629.269539999999,
            "unit": "MB",
            "th": 0.05
        }
    },
    "PP-HumanSeg-Lite": {
        "model_name": "PP-HumanSeg-Lite",
        "batch_size": 1,
        "jingdu": {
            "value": 0.369783895654616,
            "unit": "mIoU",
            "th": 0.01
        },
        "xingneng": {
            "value": 1.1800000000000002,
            "unit": "ms",
            "th": 0.05
        },
        "cpu_mem": {
            "value": 1387.02892,
            "unit": "MB",
            "th": 0.05
        },
        "gpu_mem": {
            "value": 263.0,
            "unit": "MB",
            "th": 0.05
        }
    },
    "PP-Liteseg": {
        "model_name": "PP-Liteseg",
        "batch_size": 1,
        "jingdu": {
            "value": 0.7496189272981876,
            "unit": "mIoU",
            "th": 0.01
        },
        "xingneng": {
            "value": 10.320000000000002,
            "unit": "ms",
            "th": 0.05
        },
        "cpu_mem": {
            "value": 1534.21794,
            "unit": "MB",
            "th": 0.05
        },
        "gpu_mem": {
            "value": 457.0,
            "unit": "MB",
            "th": 0.05
        }
    },
    "HRNet": {
        "model_name": "HRNet",
        "batch_size": 1,
        "jingdu": {
            "value": 0.7852055365324859,
            "unit": "mIoU",
            "th": 0.01
        },
        "xingneng": {
            "value": 37.72,
            "unit": "ms",
            "th": 0.05
        },
        "cpu_mem": {
            "value": 1538.2250000000001,
            "unit": "MB",
            "th": 0.05
        },
        "gpu_mem": {
            "value": 517.0,
            "unit": "MB",
            "th": 0.05
        }
    },
    "UNet": {
        "model_name": "UNet",
        "batch_size": 1,
        "jingdu": {
            "value": 0.6455812617332316,
            "unit": "mIoU",
            "th": 0.01
        },
        "xingneng": {
            "value": 82.26000000000002,
            "unit": "ms",
            "th": 0.05
        },
        "cpu_mem": {
            "value": 1522.3336,
            "unit": "MB",
            "th": 0.05
        },
        "gpu_mem": {
            "value": 1437.0,
            "unit": "MB",
            "th": 0.05
        }
    },
    "Deeplabv3-ResNet50": {
        "model_name": "Deeplabv3-ResNet50",
        "batch_size": 1,
        "jingdu": {
            "value": 0.7893610650102942,
            "unit": "mIoU",
            "th": 0.01
        },
        "xingneng": {
            "value": 97.97999999999999,
            "unit": "ms",
            "th": 0.05
        },
        "cpu_mem": {
            "value": 2172.31872,
            "unit": "MB",
            "th": 0.05
        },
        "gpu_mem": {
            "value": 1061.0,
            "unit": "MB",
            "th": 0.05
        }
    },
    "ERNIE_3.0-Medium": {
        "model_name": "ERNIE_3.0-Medium",
        "batch_size": 32,
        "jingdu": {
            "value": 0.6035681186283597,
            "unit": "acc",
            "th": 0.01
        },
        "xingneng": {
            "value": 33.92,
            "unit": "ms",
            "th": 0.05
        },
        "gpu_mem": {
            "value": 1187.0,
            "unit": "MB",
            "th": 0.05
        },
        "cpu_mem": {
            "value": 3021.94452,
            "unit": "MB",
            "th": 0.05
        }
    },
    "PP-MiniLM": {
        "model_name": "PP-MiniLM",
        "batch_size": 32,
        "jingdu": {
            "value": 0.5857738646895273,
            "unit": "acc",
            "th": 0.01
        },
        "xingneng": {
            "value": 34.044000000000004,
            "unit": "ms",
            "th": 0.05
        },
        "gpu_mem": {
            "value": 899.0,
            "unit": "MB",
            "th": 0.05
        },
        "cpu_mem": {
            "value": 1716.6960800000002,
            "unit": "MB",
            "th": 0.05
        }
    },
    "BERT_Base": {
        "model_name": "BERT_Base",
        "batch_size": 1,
        "jingdu": {
            "value": 0.2671497327351065,
            "unit": "acc",
            "th": 0.01
        },
        "xingneng": {
            "value": 1.886,
            "unit": "ms",
            "th": 0.05
        },
        "gpu_mem": {
            "value": 859.0,
            "unit": "MB",
            "th": 0.05
        },
        "cpu_mem": {
            "value": 3687.3656200000005,
            "unit": "MB",
            "th": 0.05
        }
    }
}
