import pytest
from pytest_assume.plugin import assume
from pytest import approx
import numpy as np
import subprocess
import re
import ast
import logging
import os
import yaml
import os.path
import platform

rec_image_shape_dict={'CRNN':'3,32,100', 'ABINet':'3,32,128', 'ViTSTR':'1,224,224' }


def metricExtraction(keyword, output):
    for line in output.split('\n'):
            if keyword in  line:
                  output_rec=line
    metric=output_rec.split(':')[-1]
    print(metric)
    return metric
          # rec_docs=output_rec_list[0].split(',')[0].strip("'")
          # rec_scores=output_rec_list[0].split(',')[1]
          # rec_scores=float(rec_scores)



class RepoInit():
      def __init__(self, repo):
         self.repo=repo
         print("This is Repo Init!")
         pid = os.getpid()
         cmd='''ps aux| grep python | grep -v %s | awk '{print $2}'| xargs kill -9; rm -rf %s; git clone https://github.com/paddlepaddle/%s.git; cd %s; python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple''' % (pid, self.repo, self.repo, self.repo)
         repo_result=subprocess.getstatusoutput(cmd)
         exit_code=repo_result[0]
         output=repo_result[1]
         assert exit_code == 0, "git clone %s failed!   log information:%s" % (self.repo, output)
         logging.info("git clone"+self.repo+"sucessfuly!" )

class RepoDataset():
      def __init__(self):
         self.config=yaml.load(open('TestCase.yaml','rb'), Loader=yaml.Loader)
         sysstr = platform.system()
         if(sysstr =="Linux"):
            print ("config Linux data_path")
            data_path=self.config["data_path"]["linux_data_path"]
            print(data_path)
            cmd='''cd PaddleOCR; rm -rf train_data; ln -s %s train_data; find configs/rec -name "*.yml" | grep -v "rec_multi_language_lite_train"  > ../models_list_rec.yaml''' % (data_path) 

         elif(sysstr == "Windows"):
            print ("config windows data_path")
            data_path=self.config["data_path"]["windows_data_path"]
            print(data_path)
            mv="ren"
            rm="del"
            cmd='''cd PaddleOCR & rd /s /q train_data & mklink /j train_data %s''' % (data_path)
         elif(sysstr == "Darwin"):
            print ("config mac data_path")
            data_path=self.config["data_path"]["mac_data_path"]
            print(data_path)
            cmd='''cd PaddleOCR; rm -rf train_data; ln -s %s train_data''' % (data_path)
         else:
            print ("Other System tasks")
            exit(1)
         print(cmd)
         repo_result=subprocess.getstatusoutput(cmd)
         exit_code=repo_result[0]
         output=repo_result[1]
         assert exit_code == 0, "configure failed!   log information:%s" % output
         logging.info("configure dataset sucessfuly!" )


def exit_check_fucntion(exit_code, output, mode, log_dir=''):
    print(output)
    assert exit_code == 0, " %s  model pretrained failed!   log information:%s" % (mode, output)
    assert 'Error' not in output, "%s  model failed!   log information:%s" % (mode, output)
    if 'ABORT!!!' in output:
         log_dir=os.path.abspath(log_dir)
         all_files=os.listdir(log_dir)
         for file in all_files:
             print (file)
             filename=os.path.join(log_dir, file)
             with open(filename) as file_obj:
                 content = file_obj.read()
                 print(content)
    assert 'ABORT!!!' not in output, "%s  model failed!   log information:%s" % (mode, output)
    logging.info("train model sucessfuly!" )

def  check_infer_metric(category, output):
     if category=='rec':
        metric=metricExtraction('result', output)
        rec_docs=metric.strip().split('\t')[0]
        rec_scores=metric.strip().split('\t')[1]
        rec_scores=float(rec_scores)

        print('rec_docs:{}'.format(rec_docs))
        print('rec_scores:{}'.format(rec_scores))

        expect_rec_docs='joint'
        expect_rec_scores=0.9999

        with assume: assert rec_docs == expect_rec_docs,\
                           "check rec_docs failed! real rec_docs is: %s,\
                            expect rec_docs is: %s" % (rec_docs, expect_rec_docs)
        with assume: assert rec_scores == approx(expect_rec_scores, abs=1e-2),\
                          "check rec_scores failed!   real rec_scores is: %s, \
                            expect rec_scores is: %s" % (rec_scores, expect_rec_scores)
        print("*************************************************************************")
     else:
        pass

def check_predict_metric(category, output):
    if category=='rec':
          for line in output.split('\n'):
                  if 'Predicts of' in  line:
                      output_rec=line
          output_rec_list=re.findall(r"\((.*?)\)", output_rec)
          print(output_rec_list)
          rec_docs=output_rec_list[0].split(',')[0].strip("'")
          rec_scores=output_rec_list[0].split(',')[1]
          rec_scores=float(rec_scores)

          print('rec_docs:{}'.format(rec_docs))
          print('rec_scores:{}'.format(rec_scores))
          expect_rec_docs='super'
          expect_rec_scores=0.9999
          with assume: assert rec_docs == expect_rec_docs,\
                           "check rec_docs failed! real rec_docs is: %s,\
                            expect rec_docs is: %s" % (rec_docs, expect_rec_docs)
          with assume: assert rec_scores == approx(expect_rec_scores, abs=1e-2),\
                          "check rec_scores failed!   real rec_scores is: %s, \
                            expect rec_scores is: %s" % (rec_scores, expect_rec_scores)
          print("*************************************************************************")
    else:
        pass

class TestOcrModelFunction():
      def __init__(self, model, yml, category): 
         self.model=model
         self.yaml=yml
         self.category=category
         self.testcase_yml=yaml.load(open('TestCase.yaml','rb'), Loader=yaml.Loader)
         self.tar_name=os.path.splitext(os.path.basename(self.testcase_yml[self.model]['eval_pretrained_model']))[0]
               
      def test_ocr_train(self, use_gpu):
          cmd='cd PaddleOCR; export CUDA_VISIBLE_DEVICES=0; sed -i s!data_lmdb_release/training!data_lmdb_release/validation!g %s; python -m paddle.distributed.launch --log_dir=log_%s  tools/train.py -c %s -o Global.use_gpu=%s Global.epoch_num=1 Global.save_epoch_step=1 Global.eval_batch_step=200 Global.print_batch_step=10 Global.save_model_dir=output/%s Train.loader.batch_size_per_card=10 Global.print_batch_step=1;' % (self.yaml,  self.model, self.yaml, use_gpu, self.model)
          if(platform.system() == "Windows"):
               cmd=cmd.replace(';','&')
               cmd=cmd.replace('sed','%sed%')
          if(platform.system() == "Darwin"):
               cmd=cmd.replace('sed -i','sed -i ""')
          print(cmd)
          detection_result = subprocess.getstatusoutput(cmd)
          exit_code = detection_result[0]
          output = detection_result[1]
          log_dir='PaddleOCR/log_'+self.model
          exit_check_fucntion(exit_code, output, 'train', log_dir)
      
      def test_ocr_get_pretrained_model(self):
          # cmd='cd PaddleOCR; wget %s; tar xf %s.tar; rm -rf *.tar; mv %s %s;' % (self.testcase_yml[self.model]['eval_pretrained_model'], self.tar_name, self.tar_name, self.model)
          cmd=self.testcase_yml['cmd']['rec']['get_pretrained_model'] % (self.testcase_yml[self.model]['eval_pretrained_model'], self.tar_name, self.tar_name, self.model)
          if(platform.system() == "Windows"):
               cmd=cmd.replace(';','&')
               cmd=cmd.replace('rm -rf', 'del')
               cmd=cmd.replace('mv','ren')
          print(cmd)
          detection_result = subprocess.getstatusoutput(cmd)
          exit_code = detection_result[0]
          output = detection_result[1]
          exit_check_fucntion(exit_code, output, 'eval')

      def test_ocr_eval(self, use_gpu):
          # cmd='cd PaddleOCR; python tools/eval.py -c %s  -o Global.use_gpu=%s Global.pretrained_model=./%s/best_accuracy' % (self.yaml, use_gpu, self.model)
          cmd=self.testcase_yml['cmd'][self.category]['eval'] % (self.yaml, use_gpu, self.model) 
          if(platform.system() == "Windows"):
               cmd=cmd.replace(';','&')
          print(cmd)
          detection_result = subprocess.getstatusoutput(cmd)
          exit_code = detection_result[0]
          output = detection_result[1]
          exit_check_fucntion(exit_code, output, 'eval')
          real_metric=metricExtraction('acc', output)
          expect_metric=self.testcase_yml[self.model]['eval_acc']
          real_metric=float(real_metric)
          expect_metric=float(expect_metric)

          with assume: assert real_metric == approx(expect_metric, abs=3e-2),\
                          "check eval_acc failed!   real eval_acc is: %s, \
                            expect eval_acc is: %s" % (real_metric, expect_metric)

      def test_ocr_rec_infer(self, use_gpu):
          # cmd='cd PaddleOCR; python tools/infer_rec.py -c %s  -o Global.use_gpu=%s Global.pretrained_model=./%s/best_accuracy Global.infer_img="./doc/imgs_words/en/word_1.png";' % (self.yaml, use_gpu, self.model)
          cmd=self.testcase_yml['cmd'][self.category]['infer'] % (self.yaml, use_gpu, self.model)
          if(platform.system() == "Windows"):
               cmd=cmd.replace(';','&')
          detection_result = subprocess.getstatusoutput(cmd)
          exit_code = detection_result[0]
          output = detection_result[1]
          exit_check_fucntion(exit_code, output, 'infer')

          check_infer_metric(self.category, output)          


      def test_ocr_export_model(self, use_gpu):
          # cmd='cd PaddleOCR; python tools/export_model.py -c %s -o Global.use_gpu=%s Global.pretrained_model=./%s/best_accuracy Global.save_inference_dir=./models_inference/%s;' % (self.yaml, use_gpu, self.model, self.model)
          cmd=self.testcase_yml['cmd'][self.category]['export_model'] % (self.yaml, use_gpu, self.model, self.model) 
          print(cmd)
          if(platform.system() == "Windows"):
               cmd=cmd.replace(';','&')
          detection_result = subprocess.getstatusoutput(cmd)
          exit_code = detection_result[0]
          output = detection_result[1]
          exit_check_fucntion(exit_code, output, 'export_model')

      def test_ocr_rec_predict(self, use_gpu, use_tensorrt, enable_mkldnn):
          model_config=yaml.load(open(os.path.join('PaddleOCR',self.yaml),'rb'), Loader=yaml.Loader)
          rec_algorithm=model_config['Architecture']['algorithm']
          print(rec_algorithm)
          if self.category=='rec':
             rec_image_shape=rec_image_shape_dict[rec_algorithm]
             rec_char_dict_path=self.testcase_yml[self.model]['rec_char_dict_path']

             print(rec_image_shape)
          # cmd='cd PaddleOCR; python tools/infer/predict_rec.py --image_dir="./doc/imgs_words_en/word_336.png" --rec_model_dir="./models_inference/"%s --rec_image_shape=%s --rec_algorithm=%s --rec_char_dict_path=%s --use_gpu=%s --use_tensorrt=%s --enable_mkldnn=%s;' % (self.model, rec_image_shape, rec_algorithm, rec_char_dict_path, use_gpu, use_tensorrt, enable_mkldnn)
             cmd=self.testcase_yml['cmd'][self.category]['predict'] % (self.model, rec_image_shape, rec_algorithm, rec_char_dict_path, use_gpu, use_tensorrt, enable_mkldnn)
          else:
             cmd=self.testcase_yml['cmd'][self.category]['predict'] % (self.model, use_gpu, use_tensorrt, enable_mkldnn)
          if(platform.system() == "Windows"):
               cmd=cmd.replace(';','&')
          detection_result = subprocess.getstatusoutput(cmd)
          print(cmd)
          exit_code = detection_result[0]
          output = detection_result[1]
          exit_check_fucntion(exit_code, output, 'predict')
          # acc
          # metricExtraction('Predicts', output)
          check_predict_metric(self.category, output)
