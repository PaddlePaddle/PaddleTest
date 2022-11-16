# encoding: utf-8
"""
执行case前: 生成yaml, 设置特殊参数, 改变监控指标
"""
import os
import sys
import json
import shutil
import logging
import tarfile
import argparse
import yaml
import wget
import numpy as np

logger = logging.getLogger("ce")
# TODO wget容易卡死, 增加超时计时器 https://blog.csdn.net/weixin_42368421/article/details/101354628


class PaddleGAN_Start(object):
    """
    自定义环境准备
    """

    def __init__(self):
        """
        初始化变量
        """
        self.qa_yaml_name = os.environ["qa_yaml_name"]
        self.rd_yaml_path = os.environ["rd_yaml_path"]
        logger.info("### self.qa_yaml_name: {}".format(self.qa_yaml_name))
        self.reponame = os.environ["reponame"]
        self.system = os.environ["system"]
        self.step = os.environ["step"]
        self.paddle_whl = os.environ["paddle_whl"]
        self.mode = os.environ["mode"]  # function or precision
        self.REPO_PATH = os.path.join(os.getcwd(), self.reponame)  # 所有和yaml相关的变量与此拼接

        self.env_dict = {}
        self.base_yaml_dict = {
            "base": "configs^edvr_m_wo_tsa.yaml",
            "single_only": "configs^lapstyle_draft.yaml",
        }

    def download_data(self, value=None):
        """
        下载数据集
        """
        # 调用函数路径已切换至PaddleGAN
        tar_name = value.split("/")[-1]
        # if os.path.exists(tar_name) and os.path.exists(tar_name.replace(".tar", "")):
        # 有end回收数据, 只判断文件夹
        if os.path.exists(tar_name.replace(".tar", "")):
            logger.info("#### already download {}".format(tar_name))
        else:
            logger.info("#### value: {}".format(value.replace(" ", "")))
            try:
                logger.info("#### start download {}".format(tar_name))
                wget.download(value.replace(" ", ""))
                logger.info("#### end download {}".format(tar_name))
                tf = tarfile.open(tar_name)
                tf.extractall(os.getcwd())
            except:
                logger.info("#### start download failed {} failed".format(value.replace(" ", "")))
        return 0

    def download_infer_tar(self, value=None):
        """
        下载预测需要的预训练模型
        """
        pass
        return 0

    def get_params(self):
        """
        获取模型输出路径
        """
        # 获取kpi 的标签
        if "singan_sr" in self.qa_yaml_name:
            self.kpi_value_eval = "scale9/D_gradient_penalty"
        elif "singan_universal" in self.qa_yaml_name or "singan_animation" in self.qa_yaml_name:
            self.kpi_value_eval = "scale6/D_gradient_penalty"
        elif "singan_finetune" in self.qa_yaml_name:
            self.kpi_value_eval = "scale1/D_gradient_penalty"
        elif (
            "firstorder_vox_mobile_256" in self.qa_yaml_name
            or "firstorder_vox_256" in self.qa_yaml_name
            or "firstorder_fashion" in self.qa_yaml_name
        ):
            self.kpi_value_eval = "perceptual"
        elif (
            "basicvsr_reds" in self.qa_yaml_name
            or "basicvsr++_vimeo90k_BD" in self.qa_yaml_name
            or "lesrcnn_psnr_x4_div2k" in self.qa_yaml_name
            or "edvr_l_wo_tsa" in self.qa_yaml_name
            or "basicvsr++_reds" in self.qa_yaml_name
            or "pan_psnr_x4_div2k" in self.qa_yaml_name
            or "iconvsr_reds" in self.qa_yaml_name
            or "edvr_l_w_tsa" in self.qa_yaml_name
            or "edvr_m_w_tsa" in self.qa_yaml_name
            or "esrgan_psnr_x4_div2k" in self.qa_yaml_name
            or "edvr_m_wo_tsa" in self.qa_yaml_name
            or "esrgan_psnr_x2_div2k" in self.qa_yaml_name
        ):
            self.kpi_value_eval = "loss_pixel"
        elif (
            "msvsr_vimeo90k_BD" in self.qa_yaml_name
            or "realsr_bicubic_noise_x4_df2k" in self.qa_yaml_name
            or "realsr_kernel_noise_x4_dped" in self.qa_yaml_name
            or "msvsr_reds" in self.qa_yaml_name
            or "esrgan_x4_div2k" in self.qa_yaml_name
        ):
            self.kpi_value_eval = "loss_pix"
        elif "drn_psnr_x4_div2k" in self.qa_yaml_name:
            self.kpi_value_eval = "loss_dual"
        elif "lapstyle_draft" in self.qa_yaml_name:
            self.kpi_value_eval = "loss_c"
        elif "mprnet_denoising" in self.qa_yaml_name or "mprnet_deraining" in self.qa_yaml_name:
            self.kpi_value_eval = "loss"
        elif "stylegan_v2_256_ffhq" in self.qa_yaml_name:
            self.kpi_value_eval = "l_d"
        elif "animeganv2_pretrain" in self.qa_yaml_name:
            self.kpi_value_eval = "init_c_loss"
        elif "cyclegan_cityscapes" in self.qa_yaml_name or "cyclegan_horse2zebra" in self.qa_yaml_name:
            self.kpi_value_eval = "G_idt_A_loss"
        elif "photopen" in self.qa_yaml_name:
            self.kpi_value_eval = "g_featloss"
        elif "starganv2_celeba_hq" in self.qa_yaml_name or "starganv2_afhq" in self.qa_yaml_name:
            self.kpi_value_eval = "G/latent_adv"
        elif "ugatit_selfie2anime_light" in self.qa_yaml_name or "ugatit_photo2cartoon" in self.qa_yaml_name:
            self.kpi_value_eval = "discriminator_loss"
        elif "animeganv2" in self.qa_yaml_name:
            self.kpi_value_eval = "d_loss"
        elif (
            "pix2pix_facades" in self.qa_yaml_name
            or "pix2pix_cityscapes_2gpus" in self.qa_yaml_name
            or "lapstyle_rev_first" in self.qa_yaml_name
            or "lapstyle_rev_second" in self.qa_yaml_name
            or "pix2pix_cityscapes" in self.qa_yaml_name
        ):
            self.kpi_value_eval = "D_fake_loss"
        else:
            logger.info("### use default kpi_value_eval loss")
            self.kpi_value_eval = "loss"
        return 0

    def prepare_env(self):
        """
        下载预训练模型, 指定路径
        """
        self.get_params()  #  准备参数
        print("###self.kpi_value_eval", self.kpi_value_eval)
        input()
        self.env_dict["kpi_value_eval"] = self.kpi_value_eval
        return 0

    def prepare_creat_yaml(self):
        """
        基于base yaml创造新的yaml
        """
        logger.info("### self.mode {}".format(self.mode))
        # 增加 function 和 precision 的选项, 只有在precision时才进行复制,function时只用base验证
        # if self.mode == "function":
        #     if os.path.exists(os.path.join("cases", self.qa_yaml_name)) is True:  # cases 是基于最原始的路径的
        #         os.remove(os.path.join("cases", self.qa_yaml_name))  # 删除已有的 precision 用 base
        #     try:
        #         shutil.copy(
        #             os.path.join("base", self.model_type + "_base.yaml"), \
        #                 os.path.join("cases", self.qa_yaml_name) + ".yaml"
        #         )
        #     except IOError as e:
        #         logger.info("Unable to copy file. %s" % e)
        #     except:
        #         logger.info("Unexpected error: {}".format(sys.exc_info()))
        # else:
        if os.path.exists(os.path.join("cases", self.qa_yaml_name) + ".yaml") is False:  # cases 是基于最原始的路径的
            logger.info("#### build new yaml :{}".format(os.path.join("cases", self.qa_yaml_name) + ".yaml"))

            if (
                "lapstyle_draft" in self.qa_yaml_name
                or "lapstyle_rev_first" in self.qa_yaml_name
                or "lapstyle_rev_second" in self.qa_yaml_name
                or "singan_finetune" in self.qa_yaml_name
                or "singan_animation" in self.qa_yaml_name
                or "singan_sr" in self.qa_yaml_name
                or "singan_universal" in self.qa_yaml_name
                or "prenet" in self.qa_yaml_name
                or "firstorder_vox_mobile_256" in self.qa_yaml_name
            ):
                source_yaml_name = self.base_yaml_dict["single_only"]
            else:
                source_yaml_name = self.base_yaml_dict["base"]
            try:
                shutil.copy(os.path.join("cases", source_yaml_name), os.path.join("cases", self.qa_yaml_name) + ".yaml")
            except IOError as e:
                logger.info("Unable to copy file. %s" % e)
            except:
                logger.info("Unexpected error:", sys.exc_info())
        return 0

    def prepare_gpu_env(self):
        """
        根据操作系统获取用gpu还是cpu
        """
        if "cpu" in self.system or "mac" in self.system:
            self.env_dict["set_cuda_flag"] = "cpu"  # 根据操作系统判断
        else:
            self.env_dict["set_cuda_flag"] = "gpu"  # 根据操作系统判断
        return 0

    def build_prepare(self):
        """
        执行准备过程
        """
        # 进入repo中
        ret = 0
        ret = self.prepare_env()
        if ret:
            logger.info("build prepare_env failed")
            return ret
        ret = self.prepare_gpu_env()
        if ret:
            logger.info("build prepare_gpu_env failed")
            return ret
        ret = self.prepare_creat_yaml()
        if ret:
            logger.info("build prepare_creat_yaml failed")
            return ret

        os.environ[self.reponame] = json.dumps(self.env_dict)
        return ret


def run():
    """
    执行入口
    """
    model = PaddleGAN_Start()
    model.build_prepare()
    return 0


if __name__ == "__main__":
    # args = None
    # model = PaddleGAN_Start(args)
    # model.build_prepare()
    run()
