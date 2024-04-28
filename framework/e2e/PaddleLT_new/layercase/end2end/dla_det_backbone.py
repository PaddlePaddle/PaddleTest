import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Normal, Constant, XavierUniform
from paddle.regularizer import L2Decay
from paddle.vision.ops import DeformConv2D

DLA_cfg = {34: ([1, 1, 1, 2, 2, 1], [16, 32, 64, 128, 256, 512]), }

class ConvNormLayer(nn.Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size,
                 stride,
                 groups=1,
                 norm_type='bn',
                 norm_decay=0.,
                 norm_groups=32,
                 use_dcn=False,
                 bias_on=False,
                 lr_scale=1.,
                 freeze_norm=False,
                 initializer=Normal(
                     mean=0., std=0.01),
                 skip_quant=False,
                 dcn_lr_scale=2.,
                 dcn_regularizer=L2Decay(0.)):
        super(ConvNormLayer, self).__init__()
        assert norm_type in ['bn', 'sync_bn', 'gn', None]

        if bias_on:
            bias_attr = ParamAttr(
                initializer=Constant(value=0.), learning_rate=lr_scale)
        else:
            bias_attr = False

        if not use_dcn:
            self.conv = nn.Conv2D(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                groups=groups,
                weight_attr=ParamAttr(
                    initializer=initializer, learning_rate=1.),
                bias_attr=bias_attr)
            if skip_quant:
                self.conv.skip_quant = True
        else:
            # in FCOS-DCN head, specifically need learning_rate and regularizer
            self.conv = DeformableConvV2(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                groups=groups,
                weight_attr=ParamAttr(
                    initializer=initializer, learning_rate=1.),
                bias_attr=True,
                lr_scale=dcn_lr_scale,
                regularizer=dcn_regularizer,
                dcn_bias_regularizer=dcn_regularizer,
                dcn_bias_lr_scale=dcn_lr_scale,
                skip_quant=skip_quant)

        norm_lr = 0. if freeze_norm else 1.
        param_attr = ParamAttr(
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay) if norm_decay is not None else None)
        bias_attr = ParamAttr(
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay) if norm_decay is not None else None)
        if norm_type in ['bn', 'sync_bn']:
            self.norm = nn.BatchNorm2D(
                ch_out, weight_attr=param_attr, bias_attr=bias_attr)
        elif norm_type == 'gn':
            self.norm = nn.GroupNorm(
                num_groups=norm_groups,
                num_channels=ch_out,
                weight_attr=param_attr,
                bias_attr=bias_attr)
        else:
            self.norm = None

    def forward(self, inputs):
        out = self.conv(inputs)
        if self.norm is not None:
            out = self.norm(out)
        return out


class BasicBlock(nn.Layer):
    def __init__(self, ch_in, ch_out, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = ConvNormLayer(
            ch_in,
            ch_out,
            filter_size=3,
            stride=stride,
            bias_on=False,
            norm_decay=None)
        self.conv2 = ConvNormLayer(
            ch_out,
            ch_out,
            filter_size=3,
            stride=1,
            bias_on=False,
            norm_decay=None)

    def forward(self, inputs, residual=None):
        if residual is None:
            residual = inputs

        out = self.conv1(inputs)
        out = F.relu(out)

        out = self.conv2(out)

        out = paddle.add(x=out, y=residual)
        out = F.relu(out)

        return out


class Root(nn.Layer):
    def __init__(self, ch_in, ch_out, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = ConvNormLayer(
            ch_in,
            ch_out,
            filter_size=1,
            stride=1,
            bias_on=False,
            norm_decay=None)
        self.residual = residual

    def forward(self, inputs):
        children = inputs
        out = self.conv(paddle.concat(inputs, axis=1))
        if self.residual:
            out = paddle.add(x=out, y=children[0])
        out = F.relu(out)

        return out


class Tree(nn.Layer):
    def __init__(self,
                 level,
                 block,
                 ch_in,
                 ch_out,
                 stride=1,
                 level_root=False,
                 root_dim=0,
                 root_kernel_size=1,
                 root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * ch_out
        if level_root:
            root_dim += ch_in
        if level == 1:
            self.tree1 = block(ch_in, ch_out, stride)
            self.tree2 = block(ch_out, ch_out, 1)
        else:
            self.tree1 = Tree(
                level - 1,
                block,
                ch_in,
                ch_out,
                stride,
                root_dim=0,
                root_kernel_size=root_kernel_size,
                root_residual=root_residual)
            self.tree2 = Tree(
                level - 1,
                block,
                ch_out,
                ch_out,
                1,
                root_dim=root_dim + ch_out,
                root_kernel_size=root_kernel_size,
                root_residual=root_residual)

        if level == 1:
            self.root = Root(root_dim, ch_out, root_kernel_size, root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.level = level
        if stride > 1:
            self.downsample = nn.MaxPool2D(stride, stride=stride)
        if ch_in != ch_out:
            self.project = ConvNormLayer(
                ch_in,
                ch_out,
                filter_size=1,
                stride=1,
                bias_on=False,
                norm_decay=None)

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.level == 1:
            x2 = self.tree2(x1)
            x = self.root([x2, x1] + children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class LayerCase(nn.Layer):
    """
    DLA, see https://arxiv.org/pdf/1707.06484.pdf

    Args:
        depth (int): DLA depth, only support 34 now.
        residual_root (bool): whether use a reidual layer in the root block
        pre_img (bool): add pre_img, only used in CenterTrack
        pre_hm (bool): add pre_hm, only used in CenterTrack
    """

    def __init__(self,
                 depth=34,
                 residual_root=False,
                 pre_img=False,
                 pre_hm=False):
        super(LayerCase, self).__init__()
        assert depth == 34, 'Only support DLA with depth of 34 now.'
        if depth == 34:
            block = BasicBlock
        levels, channels = DLA_cfg[depth]
        self.channels = channels
        self.num_levels = len(levels)

        self.base_layer = nn.Sequential(
            ConvNormLayer(
                3,
                channels[0],
                filter_size=7,
                stride=1,
                bias_on=False,
                norm_decay=None),
            nn.ReLU())
        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(
            levels[2],
            block,
            channels[1],
            channels[2],
            2,
            level_root=False,
            root_residual=residual_root)
        self.level3 = Tree(
            levels[3],
            block,
            channels[2],
            channels[3],
            2,
            level_root=True,
            root_residual=residual_root)
        self.level4 = Tree(
            levels[4],
            block,
            channels[3],
            channels[4],
            2,
            level_root=True,
            root_residual=residual_root)
        self.level5 = Tree(
            levels[5],
            block,
            channels[4],
            channels[5],
            2,
            level_root=True,
            root_residual=residual_root)

        if pre_img:
            self.pre_img_layer = nn.Sequential(
                ConvNormLayer(
                    3,
                    channels[0],
                    filter_size=7,
                    stride=1,
                    bias_on=False,
                    norm_decay=None),
                nn.ReLU())
        if pre_hm:
            self.pre_hm_layer = nn.Sequential(
                ConvNormLayer(
                    1,
                    channels[0],
                    filter_size=7,
                    stride=1,
                    bias_on=False,
                    norm_decay=None),
                nn.ReLU())
        self.pre_img = pre_img
        self.pre_hm = pre_hm

    def _make_conv_level(self, ch_in, ch_out, conv_num, stride=1):
        modules = []
        for i in range(conv_num):
            modules.extend([
                ConvNormLayer(
                    ch_in,
                    ch_out,
                    filter_size=3,
                    stride=stride if i == 0 else 1,
                    bias_on=False,
                    norm_decay=None), nn.ReLU()
            ])
            ch_in = ch_out
        return nn.Sequential(*modules)

    def forward(self, inputs):
        outs = []
        feats = self.base_layer(inputs)

        # if self.pre_img and 'pre_image' in inputs and inputs[
        #         'pre_image'] is not None:
        #     feats = feats + self.pre_img_layer(inputs['pre_image'])

        # if self.pre_hm and 'pre_hm' in inputs and inputs['pre_hm'] is not None:
        #     feats = feats + self.pre_hm_layer(inputs['pre_hm'])

        for i in range(self.num_levels):
            feats = getattr(self, 'level{}'.format(i))(feats)
            outs.append(feats)

        return outs

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
