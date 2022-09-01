
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
import allure
import filecmp
import chardet
import paddle


rec_image_shape_dict={'CRNN':'3,32,100', 'ABINet':'3,32,128', 'ViTSTR':'1,224,224', 'VisionLAN':'3,64,256' }

def metricExtraction(keyword, output):
    for line in output.split('\n'):
            if (keyword+':' in  line) and ('best_accuracy' not in line):
                  output_rec=line
                  break
    print(output_rec)
    metric=output_rec.split(':')[-1]
    print(metric)
    return metric
          # rec_docs=output_rec_list[0].split(',')[0].strip("'")
          # rec_scores=output_rec_list[0].split(',')[1]
          # rec_scores=float(rec_scores)


def platformAdapter(cmd):
    if (platform.system() == "Windows"):
            cmd=cmd.replace(';','&')
            cmd=cmd.replace('sed','%sed%')
            cmd=cmd.replace('rm -rf','rd /s /q')
            cmd=cmd.replace('export','set')
    if (platform.system() == "Darwin"):
            cmd=cmd.replace('sed -i','sed -i ""')
    return cmd
   


class RepoInit():
      def __init__(self, repo):
         self.repo=repo
         print("This is Repo Init!")
         pid = os.getpid()
         cmd='''git clone -b dygraph https://github.com/paddlepaddle/%s.git --depth 1; cd %s; python -m pip install -r requirements.txt; python -m pip install -r ppstructure/kie/requirements.txt;''' % (self.repo, self.repo)
         if(platform.system() == "Windows"):
               cmd=cmd.replace(';','&')
         repo_result=subprocess.getstatusoutput(cmd)
         exit_code=repo_result[0]
         output=repo_result[1]
         print(output)
         assert exit_code == 0, "git clone %s failed!   log information:%s" % (self.repo, output)
         logging.info("git clone"+self.repo+"sucessfuly!" )

class RepoInit3D():
      def __init__(self, repo):
         self.repo=repo
         print("This is Repo Init!")
         pid = os.getpid()
         cmd='''git clone -b develop https://github.com/paddlepaddle/%s.git --depth 1; cd %s; python -m pip install -r requirements.txt; python -m pip uninstall -y paddle3d; python -m pip install .''' % (self.repo, self.repo)
         if(platform.system() == "Windows"):
               cmd=cmd.replace(';','&')
         repo_result=subprocess.getstatusoutput(cmd)
         exit_code=repo_result[0]
         output=repo_result[1]
         print(output)
         assert exit_code == 0, "git clone %s failed!   log information:%s" % (self.repo, output)
         logging.info("git clone"+self.repo+"sucessfuly!" )

class RepoDataset3D():
      def __init__(self):
         sysstr = platform.system()
         if(sysstr =="Linux"):
            print ("config Linux data_path")
            cmd='''cd Paddle3D; rm -rf datasets; ln -s /ssd2/ce_data/Paddle3D datasets;'''

         elif(sysstr == "Windows"):
            print ("config windows data_path")
            cmd='''cd Paddle3D & rd /s /q datasets & mklink /j datasets E:\ce_data\Paddle3D'''
         elif(sysstr == "Darwin"):
            print ("config mac data_path")
            cmd='''cd PaddleOCR; rm -rf datasets; ln -s /Users/paddle/PaddleTest/ce_data/Paddle3D datasets'''
         else:
            print ("Other System tasks")
            exit(1)
         print(cmd)
         repo_result=subprocess.getstatusoutput(cmd)
         exit_code=repo_result[0]
         output=repo_result[1]
         assert exit_code == 0, "configure failed!   log information:%s" % output


class RepoInitSpeech():
      def __init__(self, repo):
         self.repo=repo
         print("This is Repo Init!")
         pid = os.getpid()
         cmd='''git clone -b r1.1 https://github.com/paddlepaddle/%s.git --depth 1; cd %s; yum update; yum install libsndfile -y;python -m pip uninstall -y paddlespeech; python -m pip install .''' % (self.repo, self.repo)
         if(platform.system() == "Windows"):
               cmd=cmd.replace(';','&')
         repo_result=subprocess.getstatusoutput(cmd)
         exit_code=repo_result[0]
         output=repo_result[1]
         assert exit_code == 0, "git clone %s failed!   log information:%s" % (self.repo, output)
         logging.info("git clone"+self.repo+"sucessfuly!" )

class RepoDatasetSpeech():
      def __init__(self):
         sysstr = platform.system()
         if(sysstr =="Linux"):
            print ("config Linux data_path")
            cmd='''cd PaddleSpeech/examples/zh_en_tts/tts3; rm -rf dump; ln -s /ssd2/ce_data/PaddleSpeech_t2s/preprocess_data/zh_en_tts3/dump dump; cd ../../csmsc/vits; rm -rf dump; ln -s /ssd2/ce_data/PaddleSpeech_t2s/preprocess_data/vits/dump dump; cd ../../../..; wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/cat.wav https://paddlespeech.bj.bcebos.com/PaddleAudio/dog.wav https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav https://paddlespeech.bj.bcebos.com/PaddleAudio/en.wav https://paddlespeech.bj.bcebos.com/datasets/single_wav/zh/test_long_audio_01.wav https://paddlespeech.bj.bcebos.com/vector/audio/85236145389.wav; echo -e "demo1 85236145389.wav \n demo2 85236145389.wav" > vec.job'''
         else:
            cmd='wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/cat.wav https://paddlespeech.bj.bcebos.com/PaddleAudio/dog.wav https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav https://paddlespeech.bj.bcebos.com/PaddleAudio/en.wav https://paddlespeech.bj.bcebos.com/datasets/single_wav/zh/test_long_audio_01.wav https://paddlespeech.bj.bcebos.com/vector/audio/85236145389.wav; echo -e "demo1 85236145389.wav \n demo2 85236145389.wav" > vec.job'
         print(cmd)
         repo_result=subprocess.getstatusoutput(cmd)
         exit_code=repo_result[0]
         output=repo_result[1]
         assert exit_code == 0, "configure failed!   log information:%s" % output



class RepoDataset():
      def __init__(self):
         self.config=yaml.load(open('TestCase.yaml','rb'), Loader=yaml.Loader)
         sysstr = platform.system()
         if(sysstr =="Linux"):
            print ("config Linux data_path")
            data_path=self.config["data_path"]["linux_data_path"]
            print(data_path)
            cmd='''cd PaddleOCR; rm -rf train_data; ln -s %s train_data;cd ..; cd PaddleDetection; rm -rf dataset; ln -s %s dataset''' % (data_path, data_path) 

         elif(sysstr == "Windows"):
            print ("config windows data_path")
            data_path=self.config["data_path"]["windows_data_path"]
            print(data_path)
            mv="ren"
            rm="del"
            cmd='''cd PaddleOCR & rd /s /q train_data & mklink /j train_data %s; cd PaddleDetection; rd /s /q dataset; mklink /j dataset %s''' % (data_path, data_path)
         elif(sysstr == "Darwin"):
            print ("config mac data_path")
            data_path=self.config["data_path"]["mac_data_path"]
            print(data_path)
            cmd='''cd PaddleOCR; rm -rf train_data; ln -s %s train_data; cd PaddleDetection; rm -rf dataset; ln -s %s dataset''' % (data_path, data_path)
         else:
            print ("Other System tasks")
            exit(1)
         print(cmd)
         repo_result=subprocess.getstatusoutput(cmd)
         exit_code=repo_result[0]
         output=repo_result[1]
         assert exit_code == 0, "configure failed!   log information:%s" % output
         logging.info("configure dataset sucessfuly!" )
         cmd ='''cd PaddleOCR; wget -P pretrain_models https://paddle-qa.bj.bcebos.com/rocm/abinet_vl_pretrained.pdparams; wget -P pretrain_models https://paddleocr.bj.bcebos.com/dygraph_v2.1/en_det/ResNet50_dcn_asf_synthtext_pretrained.pdparams'''
         cmd=platformAdapter(cmd)
         repo_result=subprocess.getstatusoutput(cmd)
         exit_code=repo_result[0]
         output=repo_result[1]
         assert exit_code == 0, "pretrain_models configure failed!   log information:%s" % output
         if (platform.system() == "Windows") or (platform.system() == 'Linux'):
            cmd='''cd PaddleOCR; sed -i '/config.enable_tensorrt_engine/i\                config.collect_shape_range_info("shape_range_info.pbtxt")' ./tools/infer/utility.py; sed -i '/use_calib_mode=False/a\                config.enable_tuned_tensorrt_dynamic_shape("shape_range_info.pbtxt", True)' ./tools/infer/utility.py'''
            cmd=platformAdapter(cmd)
            repo_result=subprocess.getstatusoutput(cmd)
            exit_code=repo_result[0]
            output=repo_result[1]
            assert exit_code == 0, "tensorRT dynamic shape configure  failed!   log information:%s" % output


def exit_check_fucntion(exit_code, output, mode, log_dir=''):
    print(output)
    if exit_code == 0:
       allure.attach(output, 'output.log', allure.attachment_type.TEXT)
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

def check_charset(file_path):
    with open(file_path, "rb") as f:
        data = f.read(4)
        charset = chardet.detect(data)['encoding']
    return charset

def allure_attach(filename, name, fileformat):
     with open(filename, mode='rb') as f:
         file_content = f.read()
     allure.attach(file_content, name=name, attachment_type=fileformat)

def allure_step(cmd, output):
    with allure.step("运行指令：{}".format(cmd)):
           pass
    # with allure.step("运行结果：{}".format(output)):
    #       pass


def readfile(filename):
    with open(filename, mode='r', encoding='utf-8') as f:
         text = f.readline()
    return text

def  check_infer_metric(category, output, dataset):
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
     elif category=='det':
        allure_attach("PaddleOCR/checkpoints/det_db/det_results/img_10.jpg", 'checkpoints/det_db/det_results/img_10.jpg', allure.attachment_type.JPG)
        allure_attach("PaddleOCR/checkpoints/det_db/predicts_db.txt", 'checkpoints/det_db/predicts_db.txt', allure.attachment_type.TEXT)
        allure_attach("./metric/predicts_db_"+dataset+".txt", "./metric/predicts_db_"+dataset+".txt", allure.attachment_type.TEXT)
        # status = filecmp.cmp("./metric/predicts_db_"+dataset+".txt", "PaddleOCR/checkpoints/det_db/predicts_db.txt")
        real_det_bbox=readfile("PaddleOCR/checkpoints/det_db/predicts_db.txt")
        expect_det_bbox=readfile("./metric/predicts_db_"+dataset+".txt")
        assert real_det_bbox==expect_det_bbox, "real det_bbox should equal expect det_bbox"
     elif category =='table':
           real_metric=metricExtraction('result', output)
           table_bbox=real_metric.split("'</html>'],")[0]
           print("table_bbox:{}".format(table_bbox))
           # with open("./metric/infer_table.txt", mode='w', encoding='utf-8') as file_obj:
           #     file_obj.write(real_metric)
           # print("table_result:{}".format(real_metric))
           # allure_attach("PaddleOCR/output/table.jpg", './output/table.jpg', allure.attachment_type.JPG)
           allure.attach(real_metric, 'real_table_result', allure.attachment_type.TEXT)
           allure_attach("./metric/infer_table.txt", "./metric/infer_table.txt", allure.attachment_type.TEXT)

           real_table=real_metric
           expect_table=readfile("./metric/infer_table.txt")
           print(len(real_table))
           print("expect_table:{}".format(expect_table))
           print(len(expect_table))

           # assert real_table==expect_table, "real table should equal expect table"
     elif category =='sr':
          allure_attach("PaddleOCR/infer_result/sr_word_52.png", 'infer_result/sr_word_52.png', allure.attachment_type.PNG)
     elif category=='kie/vi_layoutxlm':
          if os.path.exists('./output/ser/xfund_zh/res/zh_val_0_ser.jpg'):
             allure_attach("PaddleOCR/output/ser/xfund_zh/res/zh_val_0_ser.jpg", './output/output/ser/xfund_zh/res/zh_val_0_ser.jpg', allure.attachment_type.JPG)
          if os.path.exists('./output/re/xfund_zh/with_gt/zh_val_0_ser_re.jpg'):
             allure_attach("PaddleOCR/output/re/xfund_zh/with_gt/zh_val_0_ser_re.jpg", './output/re/xfund_zh/with_gt/zh_val_0_ser_re.jpg', allure.attachment_type.JPG)
     else:
           pass

def check_predict_metric(category, output, dataset):
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
    elif category =='det':
          allure_attach("PaddleOCR/inference_results/det_res_img_10.jpg", 'inference_results/det_res_img_10.jpg', allure.attachment_type.JPG)
          allure_attach("PaddleOCR/inference_results/det_results.txt", 'inference_results/det_results.txt', allure.attachment_type.TEXT)
          for line in output.split('\n'):
                  if 'img_10.jpg' in  line:
                      output_det=line
                      print(output_det)
                      break


          det_bbox=output_det.split('\t')[-1]
          det_bbox=ast.literal_eval(det_bbox)         
          print('det_bbox:{}'.format(det_bbox))
          if dataset=='icdar15':
             expect_det_bbox=[[[39.0, 88.0], [147.0, 80.0], [149.0, 103.0], [41.0, 110.0]], [[149.0, 82.0], [199.0, 79.0], [200.0, 98.0], [150.0, 101.0]], [[35.0, 54.0], [97.0, 54.0], [97.0, 78.0], [35.0, 78.0]], [[100.0, 53.0], [141.0, 53.0], [141.0, 79.0], [100.0, 79.0]], [[181.0, 54.0], [204.0, 54.0], [204.0, 73.0], [181.0, 73.0]], [[139.0, 54.0], [187.0, 50.0], [189.0, 75.0], [141.0, 79.0]], [[193.0, 29.0], [253.0, 29.0], [253.0, 48.0], [193.0, 48.0]], [[161.0, 28.0], [200.0, 28.0], [200.0, 48.0], [161.0, 48.0]], [[107.0, 21.0], [161.0, 24.0], [159.0, 49.0], [105.0, 46.0]], [[29.0, 19.0], [107.0, 19.0], [107.0, 46.0], [29.0, 46.0]]]
          else:
             expect_det_bbox= [[[42.0, 89.0], [201.0, 79.0], [202.0, 98.0], [43.0, 108.0]], [[32.0, 56.0], [206.0, 53.0], [207.0, 75.0], [32.0, 78.0]], [[18.0, 22.0], [251.0, 31.0], [250.0, 49.0], [17.0, 41.0]]]

          with assume: assert np.array(det_bbox) == approx(np.array(expect_det_bbox), abs=2), "check det_bbox failed!  \
                           real det_bbox is: %s, expect det_bbox is: %s" % (det_bbox, expect_det_bbox)
          print("*************************************************************************")
    elif category =='table':
           # real_metric=metricExtraction('result', output)
           # table_bbox=real_metric.split("]")[0]
           # pattern=re.compile('\[\[.+\]\]')
           # real_table=pattern.findall(real_metric)[0]
           # print("table_bbox:{}".format(table_bbox))
           # with open("./metric/predicts_table.txt", mode='w', encoding='utf-8') as file_obj:
           #     file_obj.write(real_metric)
           allure_attach("PaddleOCR/output/table.jpg", './output/table.jpg', allure.attachment_type.JPG)
           # allure.attach(real_metric, 'real_table_result', allure.attachment_type.TEXT)
          # allure_attach("./metric/predicts_table.txt", "./metric/predicts_table.txt", allure.attachment_type.TEXT)

           # real_table=real_metric
           # expect_metric=readfile("./metric/predicts_table.txt")
           # expect_table=pattern.findall(expect_metric)[0]
           # assert real_table==expect_table, "real table should equal expect table"
    elif category =='sr':
          allure_attach("PaddleOCR/infer_result/sr_word_52.png", 'infer_result/sr_word_52.png', allure.attachment_type.PNG)
    else:
        pass

class TestOcrModelFunction():
      def __init__(self, model='', yml='', category=''): 
          if model !='':
             self.model=model
             self.yaml=yml
             self.category=category
             self.testcase_yml=yaml.load(open('TestCase.yaml','rb'), Loader=yaml.Loader)
             self.tar_name=os.path.splitext(os.path.basename(self.testcase_yml[self.model]['eval_pretrained_model']))[0]
             self.dataset=self.testcase_yml[self.model]['dataset']

      def test_ocr_cli(self, cmd):
          cmd=platformAdapter(cmd)
          print(cmd)
          cmd_result = subprocess.getstatusoutput(cmd)
          exit_code = cmd_result[0]
          output = cmd_result[1]
          allure_step(cmd, output)
          exit_check_fucntion(exit_code, output, 'cli')

      def test_ocr_train(self, use_gpu):
          if self.category=='rec' or self.category=='rec/PP-OCRv3':
             cmd=self.testcase_yml['cmd'][self.category]['train'] % (self.yaml, self.yaml, use_gpu, self.model)
          elif self.category=='picodet/legacy_model/application/layout_analysis':
             cmd=self.testcase_yml['cmd'][self.category]['train'] % (use_gpu)
             if self.model=='picodet_lcnet_x2_5_layout':
                cmd=cmd+' --slim_config configs/picodet/legacy_model/application/layout_analysis/picodet_lcnet_x2_5_layout.yml'
          else:
             cmd=self.testcase_yml['cmd'][self.category]['train'] % (self.yaml, use_gpu, self.model)


          if(platform.system() == "Windows"):
               cmd=cmd.replace(';','&')
               cmd=cmd.replace('sed','%sed%')
               cmd=cmd.replace('export','set')
          if(platform.system() == "Darwin"):
               cmd=cmd.replace('sed -i','sed -i ""')
          print(cmd)
          detection_result = subprocess.getstatusoutput(cmd)
          exit_code = detection_result[0]
          output = detection_result[1]
          allure_step(cmd, output)
          log_dir='PaddleOCR/log_'+self.model
          exit_check_fucntion(exit_code, output, 'train', log_dir)

      def test_ocr_train_acc(self):
          # if self.category=='rec':
          if self.model=='rec_vitstr_none_ce':
              data1=getdata('log/rec/'+self.model+'_paddle.log', 'loss:', ', avg_reader_cost')
              data2=getdata('log/rec/'+self.model+'_torch.log', 'tensor\(', ', device=')
              allure.attach.file('log/rec/'+self.model+'_paddle.log', name=self.model+'_paddle.log',  attachment_type=allure.attachment_type.TEXT)
              allure.attach.file('log/rec/'+self.model+'_torch.log', name=self.model+'_torch.log', attachment_type=allure.attachment_type.TEXT)
              plot_paddle_torch_loss(data1, data2, self.model)          
              allure.attach.file('paddle_torch_train_loss.png', name='paddle_torch_train_loss.png', attachment_type=allure.attachment_type.PNG)
          elif self.model=='rec_r45_abinet':
              data1=getdata_custom('log/rec/'+self.model+'_paddle.log',  ', loss:', ', avg_reader_cost')
              data2=getdata('log/rec/'+self.model+'_torch.log', 'loss =', ',  smooth')
              
              allure.attach.file('log/rec/'+self.model+'_paddle.log', name=self.model+'_paddle.log',  attachment_type=allure.attachment_type.TEXT)
              allure.attach.file('log/rec/'+self.model+'_torch.log', name=self.model+'_torch.log', attachment_type=allure.attachment_type.TEXT)
              plot_paddle_torch_loss(data1, data2, self.model)
              allure.attach.file('paddle_torch_train_loss.png', name='paddle_torch_train_loss.png', attachment_type=allure.attachment_type.PNG)
          elif self.model=='table_master':
              data1=getdata('log/table/'+self.model+'_paddle.log', ', loss:', ', horizon_bbox_loss')
              data2=getdata('log/table/'+self.model+'_torch.log', ', loss:', ', grad_norm')
              allure.attach.file('log/table/'+self.model+'_paddle.log', name=self.model+'_paddle.log',  attachment_type=allure.attachment_type.TEXT)
              allure.attach.file('log/table/'+self.model+'_torch.log', name=self.model+'_torch.log', attachment_type=allure.attachment_type.TEXT)
              plot_paddle_torch_loss(data1, data2, self.model)
          else:
              pass

      def test_ocr_get_pretrained_model(self):
          # cmd='cd PaddleOCR; wget %s; tar xf %s.tar; rm -rf *.tar; mv %s %s;' % (self.testcase_yml[self.model]['eval_pretrained_model'], self.tar_name, self.tar_name, self.model)
          if (self.category=='table') or (self.category=='kie/vi_layoutxlm') or (self.category=='det/ch_PP-OCRv3') or (self.category=='rec/PP-OCRv3') or (self.category=='e2e'):
              cmd=self.testcase_yml['cmd'][self.category]['get_pretrained_model'] % (self.testcase_yml[self.model]['eval_pretrained_model'], self.tar_name, self.tar_name, self.model)
          elif self.category=='picodet/legacy_model/application/layout_analysis':
              cmd=self.testcase_yml['cmd'][self.category]['get_pretrained_model']
          else: 
              cmd=self.testcase_yml['cmd'][self.category]['get_pretrained_model'] % (self.testcase_yml[self.model]['eval_pretrained_model'], self.tar_name, self.model, self.model)
          
          if(platform.system() == "Windows"):
               cmd=cmd.replace(';','&')
               cmd=cmd.replace('rm -rf', 'del')
               cmd=cmd.replace('mv','ren')
          print(cmd)
          detection_result = subprocess.getstatusoutput(cmd)
          exit_code = detection_result[0]
          output = detection_result[1]
          allure_step(cmd, output)
          exit_check_fucntion(exit_code, output, 'eval')

      def test_ocr_eval(self, use_gpu):
          # cmd='cd PaddleOCR; python tools/eval.py -c %s  -o Global.use_gpu=%s Global.pretrained_model=./%s/best_accuracy' % (self.yaml, use_gpu, self.model)
          if self.category=='picodet/legacy_model/application/layout_analysis':
              cmd=self.testcase_yml['cmd'][self.category]['eval'] % (use_gpu)
              if self.model=='picodet_lcnet_x2_5_layout':
                 cmd=cmd+' --slim_config configs/picodet/legacy_model/application/layout_analysis/picodet_lcnet_x2_5_layout.yml'
          else:
              cmd=self.testcase_yml['cmd'][self.category]['eval'] % (self.yaml, use_gpu, self.model) 
          if(platform.system() == "Windows"):
               cmd=cmd.replace(';','&')
          cmd=cmd.replace('_udml.yml','.yml')
          print(cmd)
          detection_result = subprocess.getstatusoutput(cmd)
          exit_code = detection_result[0]
          output = detection_result[1]
          allure_step(cmd, output)
          exit_check_fucntion(exit_code, output, 'eval')
          '''
          if self.category=='rec' or self.category=='table':
             keyword='acc'
          elif (self.category=='det') or (self.category=='table') or (self.category=='kie/vi_layoutxlm'):
             keyword='hmean'
          elif self.category=='sr':
             keyword='psnr_avg'
          else:
             pass
      
            
          real_metric=metricExtraction(keyword, output)
          expect_metric=self.testcase_yml[self.model]['eval_'+keyword]
          
          # attach 
          body="expect_"+keyword+": "+str(expect_metric)
          allure.attach(body, 'expect_metric', allure.attachment_type.TEXT)
          body="real_"+keyword+": "+real_metric
          allure.attach(body, 'real_metric', allure.attachment_type.TEXT)
          
          # assert
          real_metric=float(real_metric)
          expect_metric=float(expect_metric)
          with assume: assert real_metric == approx(expect_metric, abs=3e-2),\
                          "check eval_acc failed!   real eval_acc is: %s, \
                            expect eval_acc is: %s" % (real_metric, expect_metric)
          '''

      def test_ocr_rec_infer(self, use_gpu):
          if self.category=='picodet/legacy_model/application/layout_analysis':
              cmd=self.testcase_yml['cmd'][self.category]['infer'] % (use_gpu)
              if self.model=='picodet_lcnet_x2_5_layout':
                 cmd=cmd+' --slim_config configs/picodet/legacy_model/application/layout_analysis/picodet_lcnet_x2_5_layout.yml'
          else:
              cmd=self.testcase_yml['cmd'][self.category]['infer'] % (self.yaml, use_gpu, self.model)
          if (self.model=='re_vi_layoutxlm_xfund_zh'):
             cmd=cmd.replace('infer_kie_token_ser','infer_kie_token_ser_re')
             cmd =cmd+' -c_ser configs/kie/vi_layoutxlm/ser_vi_layoutxlm_xfund_zh.yml -o_ser Architecture.Backbone.checkpoints=./ser_vi_layoutxlm_xfund_zh'
          cmd=cmd.replace('_udml.yml','.yml')
          if(platform.system() == "Windows"):
               cmd=cmd.replace(';','&')
         
          print(cmd)
          detection_result = subprocess.getstatusoutput(cmd)
          exit_code = detection_result[0]
          output = detection_result[1]
          allure_step(cmd, output)
          exit_check_fucntion(exit_code, output, 'infer')
          check_infer_metric(self.category, output, self.dataset)          


      def test_ocr_export_model(self, use_gpu):
          if self.category=='picodet/legacy_model/application/layout_analysis':
              cmd=self.testcase_yml['cmd'][self.category]['export_model'] % (use_gpu)
              if self.model=='picodet_lcnet_x2_5_layout':
                 cmd=cmd+' --slim_config configs/picodet/legacy_model/application/layout_analysis/picodet_lcnet_x2_5_layout.yml'
          else:
              cmd=self.testcase_yml['cmd'][self.category]['export_model'] % (self.yaml, use_gpu, self.model, self.model) 
          cmd=cmd.replace('_udml.yml','.yml')
          print(cmd)
          if(platform.system() == "Windows"):
               cmd=cmd.replace(';','&')
          detection_result = subprocess.getstatusoutput(cmd)
          exit_code = detection_result[0]
          output = detection_result[1]
          allure_step(cmd, output)
          exit_check_fucntion(exit_code, output, 'export_model')

      def test_ocr_rec_predict(self, use_gpu, use_tensorrt, enable_mkldnn):
          if self.category=='rec/PP-OCRv3':
             model_config=yaml.load(open(os.path.join('PaddleOCR',self.yaml),'rb'), Loader=yaml.Loader)
             algorithm=model_config['Architecture']['algorithm']
             rec_image_shape=rec_image_shape_dict[algorithm]
             rec_char_dict_path=self.testcase_yml[self.model]['rec_char_dict_path']

             print(rec_image_shape)
          # cmd='cd PaddleOCR; python tools/infer/predict_rec.py --image_dir="./doc/imgs_words_en/word_336.png" --rec_model_dir="./models_inference/"%s --rec_image_shape=%s --rec_algorithm=%s --rec_char_dict_path=%s --use_gpu=%s --use_tensorrt=%s --enable_mkldnn=%s;' % (self.model, rec_image_shape, rec_algorithm, rec_char_dict_path, use_gpu, use_tensorrt, enable_mkldnn)
             cmd=self.testcase_yml['cmd'][self.category]['predict'] % (self.model, rec_image_shape, algorithm, rec_char_dict_path, use_gpu, use_tensorrt, enable_mkldnn)
          elif self.category=='det/ch_PP-OCRv3':
             model_config=yaml.load(open(os.path.join('PaddleOCR',self.yaml),'rb'), Loader=yaml.Loader)
             algorithm=model_config['Architecture']['algorithm']
             cmd=self.testcase_yml['cmd'][self.category]['predict'] % (self.model, algorithm, use_gpu, use_tensorrt, enable_mkldnn)
          elif self.category=='table' or self.category=='e2e':
              cmd=self.testcase_yml['cmd'][self.category]['predict'] % (self.model, use_gpu, use_tensorrt, enable_mkldnn)
          elif self.category=='sr':
             sr_image_shape=self.testcase_yml[self.model]['sr_image_shape']
             cmd=self.testcase_yml['cmd'][self.category]['predict'] % (self.model, sr_image_shape, use_gpu, use_tensorrt, enable_mkldnn)
          elif self.category=='kie/vi_layoutxlm':
             cmd=self.testcase_yml['cmd'][self.category]['predict'] % (self.model, use_gpu, use_tensorrt, enable_mkldnn)
          elif self.category=='picodet/legacy_model/application/layout_analysis':
             if use_gpu==True:
                 use_gpu='gpu'
             else:
                 use_gpu='cpu'
             cmd=self.testcase_yml['cmd'][self.category]['predict'] % (use_gpu)
        
         
          if self.model=='SLANet':
              cmd=self.testcase_yml['cmd'][self.category]['predict_SLANet'] % (self.model, use_gpu, use_tensorrt, enable_mkldnn)

          if(platform.system() == "Windows"):
               cmd=cmd.replace(';','&')
          detection_result = subprocess.getstatusoutput(cmd)
          print(cmd)
          exit_code = detection_result[0]
          output = detection_result[1]
          allure_step(cmd, output)
          exit_check_fucntion(exit_code, output, 'predict')
          # acc
          # metricExtraction('Predicts', output)
          check_predict_metric(self.category, output, self.dataset)
      
      def test_ocr_predict_recovery(self, use_gpu):
          cmd='cd PaddleOCR; python -m pip install -r ppstructure/recovery/requirements.txt; mkdir inference && cd inference; wget https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar && tar xf en_PP-OCRv3_det_infer.tar; wget https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar && tar xf en_PP-OCRv3_rec_infer.tar; wget https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/en_ppstructure_mobile_v2.0_SLANet_infer.tar && tar xf en_ppstructure_mobile_v2.0_SLANet_infer.tar; wget https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_infer.tar && tar xf picodet_lcnet_x1_0_fgd_layout_infer.tar; cd ..; python ppstructure/predict_system.py --image_dir=./ppstructure/docs/table/1.png --det_model_dir=inference/en_PP-OCRv3_det_infer --rec_model_dir=inference/en_PP-OCRv3_rec_infer --rec_char_dict_path=./ppocr/utils/en_dict.txt --table_model_dir=inference/en_ppstructure_mobile_v2.0_SLANet_infer --table_char_dict_path=./ppocr/utils/dict/table_structure_dict.txt --layout_model_dir=inference/picodet_lcnet_x1_0_fgd_layout_infer --layout_dict_path=./ppocr/utils/dict/layout_dict/layout_publaynet_dict.txt --vis_font_path=./doc/fonts/simfang.ttf --recovery=True --save_pdf=False --output=./output/'
          print(cmd)
          if(platform.system() == "Windows"):
               cmd=cmd.replace(';','&')
          detection_result = subprocess.getstatusoutput(cmd)
          exit_code = detection_result[0]
          output = detection_result[1]
          allure_step(cmd, output)
          exit_check_fucntion(exit_code, output, 'predict_recovery')


class Test3DModelFunction():
      def __init__(self, model, yml, category):
         self.model=model
         self.yaml=yml
         self.category=category

      def test_3D_train(self, use_gpu):
          cmd='cd Paddle3D; rm -rf output; export CUDA_VISIBLE_DEVICES=0; sed -i "/iters/d" %s; sed -i "1i\iters: 200"  %s ; python -m paddle.distributed.launch --log_dir=log_%s  tools/train.py --config %s --num_workers 2 --log_interval 50 --save_interval 5000' % (self.yaml,  self.yaml, self.model, self.yaml)

          cmd=platformAdapter(cmd)
          print(cmd)
          detection_result = subprocess.getstatusoutput(cmd)
          exit_code = detection_result[0]
          output = detection_result[1]
          allure_step(cmd, output)
          log_dir='Paddle3D/log_'+self.model
          exit_check_fucntion(exit_code, output, 'train', log_dir)

      def test_3D_get_pretrained_model(self):
          if self.model=='smoke_dla34_no_dcn_kitti':
             cmd='cd Paddle3D; mkdir smoke_dla34_no_dcn_kitti; cd smoke_dla34_no_dcn_kitti; wget https://paddle3d.bj.bcebos.com/models/smoke/smoke_dla34_no_dcn_kitti/model.pdparams;'
          elif self.model=='smoke_hrnet18_no_dcn_kitti':
             cmd='cd Paddle3D; mkdir %s; cd %s; wget https://paddle3d.bj.bcebos.com/models/smoke/smoke_hrnet18_no_dcn_kitti/model.pdparams' % (self.model, self.model)

          elif self.model=='pointpillars_xyres16_kitti_car' or self.model=='pointpillars_xyres16_kitti_cyclist_pedestrian':
             cmd='cd Paddle3D; mkdir %s; cd %s; wget https://bj.bcebos.com/paddle3d/models/pointpillar/%s/model.pdparams' % (self.model, self.model, self.model)
          elif self.model=='centerpoint_pillars_016voxel_kitti':
             cmd='cd Paddle3D; mkdir %s; cd %s; wget https://bj.bcebos.com/paddle3d/models/centerpoint//centerpoint_pillars_016voxel_kitti/model.pdparams'% (self.model, self.model)
          elif self.model=='centerpoint_pillars_02voxel_nuscenes_10sweep':
             cmd='cd Paddle3D; mkdir %s; cd %s; wget https://bj.bcebos.com/paddle3d/models/centerpoint//centerpoint_pillars_02voxel_nuscenes_10_sweep/model.pdparams' % (self.model, self.model)
          elif self.model=='squeezesegv3_rangenet21_semantickitti' or self.model=='squeezesegv3_rangenet53_semantickitti': 
             cmd='cd Paddle3D; mkdir %s; cd %s; wget https://bj.bcebos.com/paddle3d/models/squeezesegv3/%s/model.pdparams' % (self.model, self.model, self.model)
          if(platform.system() == "Windows"):
               cmd=cmd.replace(';','&')
               cmd=cmd.replace('rm -rf', 'del')
               cmd=cmd.replace('mv','ren')
          print(cmd)
          detection_result = subprocess.getstatusoutput(cmd)
          exit_code = detection_result[0]
          output = detection_result[1]
          allure_step(cmd, output)
          exit_check_fucntion(exit_code, output, 'eval')

      def test_3D_eval(self, use_gpu):
          cmd='cd Paddle3D; python tools/evaluate.py --config %s --num_workers 2 --model %s/model.pdparams' % (self.yaml,  self.model)
          if(platform.system() == "Windows"):
               cmd=cmd.replace(';','&')
          print(cmd)
          if (self.model=='smoke_dla34_no_dcn_kitti') or (self.model=='smoke_hrnet18_no_dcn_kitti') or (self.model=='centerpoint_pillars_016voxel_kitti') or (self.model=='centerpoint_pillars_02voxel_nuscenes_10sweep'): 
               cmd='echo "not supported for eval when bs >1"' 
          detection_result = subprocess.getstatusoutput(cmd)
          exit_code = detection_result[0]
          output = detection_result[1]
          allure_step(cmd, output)
          exit_check_fucntion(exit_code, output, 'eval')

      def test_3D_eval_bs1(self, use_gpu):
          cmd='cd Paddle3D; python tools/evaluate.py --config %s --num_workers 2 --model %s/model.pdparams --batch_size 1' % (self.yaml,  self.model)
          if(platform.system() == "Windows"):
               cmd=cmd.replace(';','&')
          print(cmd)
          detection_result = subprocess.getstatusoutput(cmd)
          exit_code = detection_result[0]
          output = detection_result[1]
          allure_step(cmd, output)
          exit_check_fucntion(exit_code, output, 'eval')
                                                                                
      def test_3D_export_model(self, use_gpu):
          if self.model=='squeezesegv3_rangenet21_semantickitti' or self.model=='squeezesegv3_rangenet53_semantickitti':
            cmd='cd Paddle3D; python tools/export.py --config %s --model %s/model.pdparams --input_shape 64 1024 --save_dir ./exported_model/%s' % (self.yaml, self.model, self.model)
          else:
            cmd='cd Paddle3D; python tools/export.py --config %s --model %s/model.pdparams --save_dir ./exported_model/%s' % (self.yaml, self.model, self.model)

          print(cmd)
          if(platform.system() == "Windows"):
               cmd=cmd.replace(';','&')
          detection_result = subprocess.getstatusoutput(cmd)
          exit_code = detection_result[0]
          output = detection_result[1]
          allure_step(cmd, output)
          exit_check_fucntion(exit_code, output, 'export_model') 

      def test_3D_predict_python(self, use_gpu, use_trt):
          infer_image='datasets/KITTI/training/velodyne/000000.bin'
          #if self.model=='pointpillars_xyres16_kitti_car':
          #    infer_image='datasets/KITTI/kitti_train_gt_database/Car/1000_Car_0.bin' 
          # elif  self.model=='pointpillars_xyres16_kitti_cyclist_pedestrian':
          #    infer_image='datasets/KITTI/kitti_train_gt_database/Cyclist/100_Cyclist_0.bin'
 
          if self.model=='smoke_dla34_no_dcn_kitti' or self.model=='smoke_hrnet18_no_dcn_kitti':
             infer_image='datasets/KITTI/training/image_2/000000.png'
             cmd='cd Paddle3D; python deploy/smoke/python/infer.py --model_file exported_model/%s/inference.pdmodel --params_file exported_model/%s/inference.pdiparams --image %s --use_gpu' % (self.model, self.model, infer_image)
             if (use_trt==True):
                cmd='cd Paddle3D; python deploy/smoke/python/infer.py --model_file exported_model/%s/inference.pdmodel --params_file exported_model/%s/inference.pdiparams --image %s --collect_dynamic_shape_info --dynamic_shape_file %s/shape_info.txt; python deploy/smoke/python/infer.py --model_file exported_model/%s/inference.pdmodel --params_file exported_model/%s/inference.pdiparams --image %s --use_gpu --use_trt --dynamic_shape_file %s/shape_info.txt;' % (self.model, self.model, infer_image, self.model, self.model, self.model, infer_image, self.model)   
             if (paddle.is_compiled_with_cuda()==False):
                cmd='cd Paddle3D; python deploy/smoke/python/infer.py --model_file exported_model/%s/inference.pdmodel --params_file exported_model/%s/inference.pdiparams --image %s' % (self.model, self.model, infer_image)

          elif self.model=='pointpillars_xyres16_kitti_car':
             cmd='cd Paddle3D; python deploy/pointpillars/python/infer.py --model_file exported_model/%s/pointpillars.pdmodel --params_file exported_model/%s/pointpillars.pdiparams --lidar_file %s --point_cloud_range 0 -39.68 -3 69.12 39.68 1 --voxel_size .16 .16 4 --max_points_in_voxel 32  --max_voxel_num 40000' % (self.model, self.model, infer_image)
             if (paddle.is_compiled_with_cuda()==False):
                cmd='cd Paddle3D; sed -i "/config.enable_use_gpu/d" deploy/%s/python/infer.py; python deploy/pointpillars/python/infer.py --model_file exported_model/%s/pointpillars.pdmodel --params_file exported_model/%s/pointpillars.pdiparams --lidar_file %s --point_cloud_range 0 -39.68 -3 69.12 39.68 1 --voxel_size .16 .16 4 --max_points_in_voxel 32  --max_voxel_num 40000' % (self.category, self.model, self.model, infer_image)

          elif self.model=='pointpillars_xyres16_kitti_cyclist_pedestrian':
             cmd='cd Paddle3D; python deploy/pointpillars/python/infer.py --model_file exported_model/%s/pointpillars.pdmodel --params_file exported_model/%s/pointpillars.pdiparams --lidar_file %s --point_cloud_range 0 -19.84 -2.5 47.36 19.84 0.5 --voxel_size .16 .16 3 --max_points_in_voxel 100 --max_voxel_num 12000' % (self.model, self.model, infer_image)
             if (paddle.is_compiled_with_cuda()==False):
                cmd='cd Paddle3D; sed -i "/config.enable_use_gpu/d" deploy/%s/python/infer.py; python deploy/pointpillars/python/infer.py --model_file exported_model/%s/pointpillars.pdmodel --params_file exported_model/%s/pointpillars.pdiparams --lidar_file %s --point_cloud_range 0 -19.84 -2.5 47.36 19.84 0.5 --voxel_size .16 .16 3 --max_points_in_voxel 100 --max_voxel_num 12000' % (self.category, self.model, self.model, infer_image)

          elif self.model=='centerpoint_pillars_016voxel_kitti' or self.model=='centerpoint_pillars_02voxel_nuscenes_10sweep':
             cmd='cd Paddle3D; python deploy/centerpoint/python/infer.py --model_file exported_model/%s/centerpoint.pdmodel --params_file exported_model/%s/centerpoint.pdiparams --lidar_file %s --num_point_dim 4' % (self.model, self.model, infer_image)
             if (paddle.is_compiled_with_cuda()==False):
                cmd='cd Paddle3D; sed -i "/config.enable_use_gpu/d" deploy/%s/python/infer.py; python deploy/centerpoint/python/infer.py --model_file exported_model/%s/centerpoint.pdmodel --params_file exported_model/%s/centerpoint.pdiparams --lidar_file %s --num_point_dim 4' % (self.category, self.model, self.model, infer_image)

          else:
             cmd='echo "not supported"'
           
          if(platform.system() == "Windows"):
               cmd=cmd.replace(';','&')
          detection_result = subprocess.getstatusoutput(cmd)
          print(cmd)
          exit_code = detection_result[0]
          output = detection_result[1]
          allure_step(cmd, output)
          exit_check_fucntion(exit_code, output, 'predict')


class TestSpeechModelFunction():
      def __init__(self, model=''):
         self.model=model
         self.testcase_yml=yaml.load(open('TestCaseSpeech.yaml','rb'), Loader=yaml.Loader)

      def test_speech_cli(self, cmd):
          cmd=platformAdapter(cmd)
          print(cmd)
          cmd_result = subprocess.getstatusoutput(cmd)
          exit_code = cmd_result[0]
          output = cmd_result[1]
          allure_step(cmd, output)
          exit_check_fucntion(exit_code, output, 'cli')

      def test_speech_get_pretrained_model(self):
          cmd=self.testcase_yml[self.model]['get_pretrained_model']    
          cmd=platformAdapter(cmd)
          print(cmd)
          detection_result = subprocess.getstatusoutput(cmd)
          exit_code = detection_result[0]
          output = detection_result[1]
          allure_step(cmd, output)
          exit_check_fucntion(exit_code, output, 'get_pretrained_model')

      def test_speech_train(self):
          cmd=self.testcase_yml[self.model]['train'] 
          cmd=platformAdapter(cmd)
          print(cmd)
          detection_result = subprocess.getstatusoutput(cmd)
          exit_code = detection_result[0]
          output = detection_result[1]
          allure_step(cmd, output)
          exit_check_fucntion(exit_code, output, 'train')

      def test_speech_synthesize_e2e(self):
          cmd=self.testcase_yml[self.model]['synthesize_e2e'] 
          cmd=platformAdapter(cmd)
          print(cmd)
          detection_result = subprocess.getstatusoutput(cmd)
          exit_code = detection_result[0]
          output = detection_result[1]
          allure_step(cmd, output)
          exit_check_fucntion(exit_code, output, 'synthesize_e2e')
