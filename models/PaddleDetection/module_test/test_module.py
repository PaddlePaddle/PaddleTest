import paddle
import os
import sys
import wget

class Test(object):
    def __init__(self, cfg):
        paddle.seed(33)
        self.net = cfg.net
        self.data = cfg.data
        self.label = cfg.label
        self.module_name = cfg.module_name
        # some module has more than one input data
        if hasattr(cfg, 'input'):
            self.input = cfg.input
        self.type = ""
        # some module has more than one predict result
        self.predicts_module = ["SOLOv2Head", "SimpleConvHead", "S2ANetHead", "RetinaHead", "PPYOLOERHead", "PPYOLOEHead", "PPYOLOEContrastHead", "PicoHeadV2", "PicoHead", "PETRHead", "OTAVFLHead", "OTAHead", "LDGFLHead", "GFLHead", "FCOSRHead", "FCOSHead", "FaceHead", "DETRHead", "DeformableDETRHead"]
    
    def backward_test(self):
        opt = paddle.optimizer.SGD(learning_rate=0.01, parameters=self.net.parameters())
        final_loss = 0
        for epoch in range(5):
            if self.module_name == "SparseRCNNHead" or self.module_name == "FaceHead" or self.module_name == "CenterTrackHead" or self.module_name == "CenterNetHead" or self.module_name == "RoIAlign":
                predicts = self.net(self.data, **self.input)
            elif self.module_name == "DETRHead" or self.module_name == "DeformableDETRHead":
                predicts = self.net(self.input, self.data)
            else:
                predicts = self.net(self.data)
            if self.module_name in self.predicts_module:
                predicts = predicts[1]
            elif self.module_name == "CenterTrackHead":
                predicts = predicts["bboxes"]
            elif self.module_name == "CenterNetHead":
                predicts = predicts["heatmap"]
            loss_sum = 0
            for i in range(len(predicts)):
                loss = paddle.nn.functional.square_error_cost(predicts[i], self.label[i])
                avg_loss = paddle.mean(loss)
                loss_sum += avg_loss
            print("loss_sum:{}".format(loss_sum))
            final_loss = loss_sum
            loss_sum.backward()
            opt.step()
            opt.clear_grad()
        print("final_loss:{}".format(final_loss))
        self.type = "backward"
        #paddle.save(final_loss, "{}_{}.pdparams".format(self.module_name, self.type))
        self.check_result(final_loss)
        return final_loss

    def forward_test(self):
        if self.module_name == "SparseRCNNHead" or self.module_name == "FaceHead" or self.module_name == "CenterTrackHead" or self.module_name == "CenterNetHead" or self.module_name == "MaskHead" or self.module_name == "PETRHead" or self.module_name == "RoIAlign":
            predicts = self.net(self.data, **self.input)
        elif self.module_name == "DETRHead" or self.module_name == "DeformableDETRHead":
            predicts = self.net(self.input, self.data)
        else:
            predicts = self.net(self.data)
        if self.module_name in self.predicts_module:
            predicts = predicts[1]
        elif self.module_name == "CenterTrackHead":
            predicts = predicts["bboxes"]
        elif self.module_name == "CenterNetHead":
                predicts = predicts["heatmap"]
        print('predicts:{}'.format(predicts))
        self.type = "forward"
        #paddle.save(predicts, "{}_{}.pdparams".format(self.module_name, self.type))
        self.check_result(predicts)
        return predicts
    
    def compare(self, result, standard):
        compare_equal = True
        if isinstance(result, list):
            tensor_num = len(result)
            for i in range(0, tensor_num):
                allclose_tensor = paddle.allclose(result[i], standard[i], rtol=1e-05, atol=1e-08)              
                allclose_bool = bool(allclose_tensor.numpy())
                compare_equal = compare_equal and allclose_bool
        else:
            allclose_tensor = paddle.allclose(result, standard, rtol=1e-05, atol=1e-08)
            allclose_bool = bool(allclose_tensor.numpy())
            compare_equal = compare_equal and allclose_bool
        return compare_equal

    def check_result(self, result):
        if not os.path.exists("standard_result"):
            wget.download("https://paddle-qa.bj.bcebos.com/PaddleDetection/standard_result.zip")
            os.system("unzip -q standard_result.zip")        
        standard_value = paddle.load('./standard_result/{}_{}.pdparams'.format(self.module_name, self.type))
        print('standard_value:{}'.format(standard_value))
        compare_res = self.compare(result, standard_value)
        if compare_res:
            print("{}_{} test success!".format(self.module_name, self.type))
        else:
            print("{}_{} test failed!".format(self.module_name, self.type))
