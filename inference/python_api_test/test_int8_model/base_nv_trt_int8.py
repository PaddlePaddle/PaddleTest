"""
nv_trt_int8 base
"""

nv_trt_int8 = {
    "PPYOLOE_PLUS": {
        "model_name": "PPYOLOE_PLUS",
        "batch_size": 1,
        "jingdu": {
            "value": 0.5584375545585372,
            "unit": "mAP",
            "th": 0.01
        },
        "xingneng": {
            "value": 3.8200000000000003,
            "unit": "ms",
            "th": 0.05
        },
        "cpu_mem": {
            "value": 2447.5328,
            "unit": "MB",
            "th": 0.05
        },
        "gpu_mem": {
            "value": 637.0,
            "unit": "MB",
            "th": 0.05
        }
    },
    "PicoDet": {
        "model_name": "PicoDet",
        "batch_size": 1,
        "jingdu": {
            "value": 0.3561621621721951,
            "unit": "mAP",
            "th": 0.01
        },
        "xingneng": {
            "value": 1.4,
            "unit": "ms",
            "th": 0.05
        },
        "cpu_mem": {
            "value": 2416.29454,
            "unit": "MB",
            "th": 0.05
        },
        "gpu_mem": {
            "value": 607.0,
            "unit": "MB",
            "th": 0.05
        }
    },
    "YOLOv5s": {
        "model_name": "YOLOv5s",
        "batch_size": 1,
        "jingdu": {
            "value": 0.4631868442433045,
            "unit": "mAP",
            "th": 0.01
        },
        "xingneng": {
            "value": 3.5200000000000005,
            "unit": "ms",
            "th": 0.05
        },
        "cpu_mem": {
            "value": 1342.1921799999998,
            "unit": "MB",
            "th": 0.05
        },
        "gpu_mem": {
            "value": 297.0,
            "unit": "MB",
            "th": 0.05
        }
    },
    "YOLOv6s": {
        "model_name": "YOLOv6s",
        "batch_size": 1,
        "jingdu": {
            "value": 0.5794493709431752,
            "unit": "mAP",
            "th": 0.01
        },
        "xingneng": {
            "value": 2.2399999999999998,
            "unit": "ms",
            "th": 0.05
        },
        "cpu_mem": {
            "value": 1320.4117199999998,
            "unit": "MB",
            "th": 0.05
        },
        "gpu_mem": {
            "value": 297.0,
            "unit": "MB",
            "th": 0.05
        }
    },
    "YOLOv7": {
        "model_name": "YOLOv7",
        "batch_size": 1,
        "jingdu": {
            "value": 0.6047272452708244,
            "unit": "mAP",
            "th": 0.01
        },
        "xingneng": {
            "value": 5.9,
            "unit": "ms",
            "th": 0.05
        },
        "cpu_mem": {
            "value": 1347.91564,
            "unit": "MB",
            "th": 0.05
        },
        "gpu_mem": {
            "value": 371.0,
            "unit": "MB",
            "th": 0.05
        }
    },
    "ResNet_vd": {
        "model_name": "ResNet_vd",
        "batch_size": 1,
        "jingdu": {
            "value": 0.7754245754245754,
            "unit": "acc",
            "th": 0.01
        },
        "xingneng": {
            "value": 0.8,
            "unit": "ms",
            "th": 0.05
        },
        "gpu_mem": {
            "value": 1307.0,
            "unit": "MB",
            "th": 0.05
        },
        "cpu_mem": {
            "value": 4865.91716,
            "unit": "MB",
            "th": 0.05
        }
    },
    "MobileNetV3_large": {
        "model_name": "MobileNetV3_large",
        "batch_size": 1,
        "jingdu": {
            "value": 0.3354645354645355,
            "unit": "acc",
            "th": 0.01
        },
        "xingneng": {
            "value": 0.6,
            "unit": "ms",
            "th": 0.05
        },
        "gpu_mem": {
            "value": 1289.0,
            "unit": "MB",
            "th": 0.05
        },
        "cpu_mem": {
            "value": 4670.06094,
            "unit": "MB",
            "th": 0.05
        }
    },
    "PPLCNetV2": {
        "model_name": "PPLCNetV2",
        "batch_size": 1,
        "jingdu": {
            "value": 0.7572427572427572,
            "unit": "acc",
            "th": 0.01
        },
        "xingneng": {
            "value": 0.4,
            "unit": "ms",
            "th": 0.05
        },
        "gpu_mem": {
            "value": 1287.0,
            "unit": "MB",
            "th": 0.05
        },
        "cpu_mem": {
            "value": 4644.83126,
            "unit": "MB",
            "th": 0.05
        }
    },
    "PPHGNet_tiny": {
        "model_name": "PPHGNet_tiny",
        "batch_size": 1,
        "jingdu": {
            "value": 0.8041958041958042,
            "unit": "acc",
            "th": 0.01
        },
        "xingneng": {
            "value": 0.8,
            "unit": "ms",
            "th": 0.05
        },
        "gpu_mem": {
            "value": 1299.0,
            "unit": "MB",
            "th": 0.05
        },
        "cpu_mem": {
            "value": 4732.16798,
            "unit": "MB",
            "th": 0.05
        }
    },
    "EfficientNetB0": {
        "model_name": "EfficientNetB0",
        "batch_size": 1,
        "jingdu": {
            "value": 0.26073926073926074,
            "unit": "acc",
            "th": 0.01
        },
        "xingneng": {
            "value": 0.9,
            "unit": "ms",
            "th": 0.05
        },
        "gpu_mem": {
            "value": 1289.0,
            "unit": "MB",
            "th": 0.05
        },
        "cpu_mem": {
            "value": 4699.13048,
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
            "value": 0.9,
            "unit": "ms",
            "th": 0.05
        },
        "cpu_mem": {
            "value": 1379.22654,
            "unit": "MB",
            "th": 0.05
        },
        "gpu_mem": {
            "value": 257.8,
            "unit": "MB",
            "th": 0.05
        }
    },
    "PP-Liteseg": {
        "model_name": "PP-Liteseg",
        "batch_size": 1,
        "jingdu": {
            "value": 0.7402814977550732,
            "unit": "mIoU",
            "th": 0.01
        },
        "xingneng": {
            "value": 11.52,
            "unit": "ms",
            "th": 0.05
        },
        "cpu_mem": {
            "value": 1529.8281200000001,
            "unit": "MB",
            "th": 0.05
        },
        "gpu_mem": {
            "value": 543.0,
            "unit": "MB",
            "th": 0.05
        }
    },
    "HRNet": {
        "model_name": "HRNet",
        "batch_size": 1,
        "jingdu": {
            "value": 0.7749321005466953,
            "unit": "mIoU",
            "th": 0.01
        },
        "xingneng": {
            "value": 27.46,
            "unit": "ms",
            "th": 0.05
        },
        "cpu_mem": {
            "value": 1528.0820200000003,
            "unit": "MB",
            "th": 0.05
        },
        "gpu_mem": {
            "value": 627.0,
            "unit": "MB",
            "th": 0.05
        }
    },
    "UNet": {
        "model_name": "UNet",
        "batch_size": 1,
        "jingdu": {
            "value": 0.6281644433416818,
            "unit": "mIoU",
            "th": 0.01
        },
        "xingneng": {
            "value": 41.059999999999995,
            "unit": "ms",
            "th": 0.05
        },
        "cpu_mem": {
            "value": 1541.5281200000002,
            "unit": "MB",
            "th": 0.05
        },
        "gpu_mem": {
            "value": 983.0,
            "unit": "MB",
            "th": 0.05
        }
    },
    "Deeplabv3-ResNet50": {
        "model_name": "Deeplabv3-ResNet50",
        "batch_size": 1,
        "jingdu": {
            "value": 0.7791989132464726,
            "unit": "mIoU",
            "th": 0.01
        },
        "xingneng": {
            "value": 41.84,
            "unit": "ms",
            "th": 0.05
        },
        "cpu_mem": {
            "value": 1524.17972,
            "unit": "MB",
            "th": 0.05
        },
        "gpu_mem": {
            "value": 617.0,
            "unit": "MB",
            "th": 0.05
        }
    },
    "ERNIE_3.0-Medium": {
        "model_name": "ERNIE_3.0-Medium",
        "batch_size": 32,
        "jingdu": {
            "value": 0.6533827618164968,
            "unit": "acc",
            "th": 0.01
        },
        "xingneng": {
            "value": 16.442,
            "unit": "ms",
            "th": 0.05
        },
        "gpu_mem": {
            "value": 1421.0,
            "unit": "MB",
            "th": 0.05
        },
        "cpu_mem": {
            "value": 3492.01642,
            "unit": "MB",
            "th": 0.05
        }
    },
    "PP-MiniLM": {
        "model_name": "PP-MiniLM",
        "batch_size": 32,
        "jingdu": {
            "value": 0.6607506950880444,
            "unit": "acc",
            "th": 0.01
        },
        "xingneng": {
            "value": 20.012,
            "unit": "ms",
            "th": 0.05
        },
        "gpu_mem": {
            "value": 1298.2,
            "unit": "MB",
            "th": 0.05
        },
        "cpu_mem": {
            "value": 3492.9086,
            "unit": "MB",
            "th": 0.05
        }
    },
    "BERT_Base": {
        "model_name": "BERT_Base",
        "batch_size": 1,
        "jingdu": {
            "value": 0.0,
            "unit": "acc",
            "th": 0.01
        },
        "xingneng": {
            "value": 2.9800000000000004,
            "unit": "ms",
            "th": 0.05
        },
        "gpu_mem": {
            "value": 1091.0,
            "unit": "MB",
            "th": 0.05
        },
        "cpu_mem": {
            "value": 3834.2695200000003,
            "unit": "MB",
            "th": 0.05
        }
    }
}
