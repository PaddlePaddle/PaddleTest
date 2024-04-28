import numpy as np
import paddle.nn as nn

from layercase.end2end.resnet_det_backbone import LayerCase as ResNet
from layercase.end2end.resnet_det_backbone import Blocks, BasicBlock, BottleNeck, NameAdapter

__all__ = ['SENet', 'SERes5Head']


class LayerCase(ResNet):
    __shared__ = ['norm_type']

    def __init__(self,
                 depth=50,
                 variant='b',
                 lr_mult_list=[1.0, 1.0, 1.0, 1.0],
                 groups=1,
                 base_width=64,
                 norm_type='bn',
                 norm_decay=0,
                 freeze_norm=True,
                 freeze_at=0,
                 return_idx=[0, 1, 2, 3],
                 dcn_v2_stages=[-1],
                 std_senet=True,
                 num_stages=4):
        """
        Squeeze-and-Excitation Networks, see https://arxiv.org/abs/1709.01507
        
        Args:
            depth (int): SENet depth, should be 50, 101, 152
            variant (str): ResNet variant, supports 'a', 'b', 'c', 'd' currently
            lr_mult_list (list): learning rate ratio of different resnet stages(2,3,4,5),
                                 lower learning rate ratio is need for pretrained model 
                                 got using distillation(default as [1.0, 1.0, 1.0, 1.0]).
            groups (int): group convolution cardinality
            base_width (int): base width of each group convolution
            norm_type (str): normalization type, 'bn', 'sync_bn' or 'affine_channel'
            norm_decay (float): weight decay for normalization layer weights
            freeze_norm (bool): freeze normalization layers
            freeze_at (int): freeze the backbone at which stage
            return_idx (list): index of the stages whose feature maps are returned
            dcn_v2_stages (list): index of stages who select deformable conv v2
            std_senet (bool): whether use senet, default True
            num_stages (int): total num of stages
        """

        super(LayerCase, self).__init__(
            depth=depth,
            variant=variant,
            lr_mult_list=lr_mult_list,
            ch_in=128,
            groups=groups,
            base_width=base_width,
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            freeze_at=freeze_at,
            return_idx=return_idx,
            dcn_v2_stages=dcn_v2_stages,
            std_senet=std_senet,
            num_stages=num_stages)


class SERes5Head(nn.Layer):
    def __init__(self,
                 depth=50,
                 variant='b',
                 lr_mult=1.0,
                 groups=1,
                 base_width=64,
                 norm_type='bn',
                 norm_decay=0,
                 dcn_v2=False,
                 freeze_norm=False,
                 std_senet=True):
        """
        SERes5Head layer

        Args:
            depth (int): SENet depth, should be 50, 101, 152
            variant (str): ResNet variant, supports 'a', 'b', 'c', 'd' currently
            lr_mult (list): learning rate ratio of SERes5Head, default as 1.0.
            groups (int): group convolution cardinality
            base_width (int): base width of each group convolution
            norm_type (str): normalization type, 'bn', 'sync_bn' or 'affine_channel'
            norm_decay (float): weight decay for normalization layer weights
            dcn_v2_stages (list): index of stages who select deformable conv v2
            std_senet (bool): whether use senet, default True
            
        """
        super(SERes5Head, self).__init__()
        ch_out = 512
        ch_in = 256 if depth < 50 else 1024
        na = NameAdapter(self)
        block = BottleNeck if depth >= 50 else BasicBlock
        self.res5 = Blocks(
            block,
            ch_in,
            ch_out,
            count=3,
            name_adapter=na,
            stage_num=5,
            variant=variant,
            groups=groups,
            base_width=base_width,
            lr=lr_mult,
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            dcn_v2=dcn_v2,
            std_senet=std_senet)
        self.ch_out = ch_out * block.expansion

    def forward(self, roi_feat):
        y = self.res5(roi_feat)
        return y


def create_tensor_inputs():
    inputs = (
        paddle.rand(shape=[4, 3, 224, 224], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[4, 3, 224, 224]).astype('float32'),
    )
    return inputs
