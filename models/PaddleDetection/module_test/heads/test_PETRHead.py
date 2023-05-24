from __future__ import absolute_import
import paddle
import os
import sys

# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 3)))
sys.path.insert(0, parent_path)
from ppdet.modeling.heads import PETRHead
from module.test_module import Test
from ppdet.modeling.transformers.petr_transformer import PETRTransformer
from ppdet.modeling.transformers.position_encoding import PositionEmbedding
from ppdet.modeling.transformers.petr_transformer import TransformerEncoder, TransformerEncoderLayer, PETR_TransformerDecoder, PETR_TransformerDecoderLayer
from ppdet.modeling.losses.focal_loss import Weighted_FocalLoss

class Config:
    def __init__(self):
        paddle.seed(33)
        self.module_name = "PETRHead"
        weighted_focalloss = Weighted_FocalLoss()
        transformerdecoderlayer = PETR_TransformerDecoderLayer(d_model=256)
        transformerdecoder = PETR_TransformerDecoder(decoder_layer=transformerdecoderlayer, num_layers=1)
        transformerencoderlayer = TransformerEncoderLayer(d_model=256, dim_feedforward=16)
        transformerencoder = TransformerEncoder(encoder_layer=transformerencoderlayer, num_layers=1)
        petrtransformer = PETRTransformer(encoder=transformerencoder, decoder=transformerdecoder)
        positionembedding = PositionEmbedding()
        self.net = PETRHead(in_channels=256, num_classes=3, num_query=100, num_kpt_fcs=2, num_keypoints=17, transformer=petrtransformer, sync_cls_avg_factor=True, positional_encoding=positionembedding, loss_cls=weighted_focalloss, loss_kpt='L1Loss', loss_oks='OKSLoss', loss_hm='CenterFocalLoss', with_kpt_refine=False, assigner='PoseHungarianAssigner', sampler='PseudoSampler')
        self.net.eval()
        feat1 = paddle.rand(shape=[1, 256, 32, 32])
        feat2 = paddle.rand(shape=[1, 256, 16, 16])
        feat3 = paddle.rand(shape=[1, 256, 8, 8])
        self.data = [feat1, feat2, feat3]
        label1 = paddle.rand(shape=[4, 1344, 3])
        label2 = paddle.rand(shape=[4, 1344, 4])
        label3 = paddle.rand(shape=[1344, 1])
        self.label = [label1, label2, label3]

cfg = Config()
test = Test(cfg)
test.forward_test()
test.backward_test()
