echo "============================================================"
echo "PP-DataClean test start！！！"
echo "============================================================"
bug=0
cd ../

easydata --model image_orientation --input ./demo/image_orientation/1.jpg --device gpu #>../tests/test_PPDC1.log
if [ $? -ne 0 ];
then
    bug=`expr ${bug} + 1`;
fi

easydata --model image_orientation --input ./demo/image_orientation/ --device gpu #>../tests/test_PPDC2.log
if [ $? -ne 0 ];
then
    bug=`expr ${bug} + 1`;
fi

easydata --model image_orientation --input ./demo/image_orientation/ --device gpu --thresh 0.93 #>../tests/test_PPDC3.log
if [ $? -ne 0 ];
then
    bug=`expr ${bug} + 1`;
fi

easydata --model clarity_assessment --input ./demo/clarity_assessment/ --device gpu #>../tests/test_PPDC4.log
if [ $? -ne 0 ];
then
    bug=`expr ${bug} + 1`;
fi

easydata --model code_exists --input ./demo/code_exists/ --device gpu #>../tests/test_PPDC5.log
if [ $? -ne 0 ];
  then
    bug=`expr ${bug} + 1`;
fi

cp ./tests/test_PPDC.py ./
python3.7  test_PPDC.py
if [ $? -ne 0 ];
  then
    bug=`expr ${bug} + 1`;
fi

echo "============================================================"
echo "PP-DataAug test start！！！"
echo "============================================================"
easydata --model ppdataaug --ori_data_dir demo/clas_data/ --label_file demo/clas_data/train_list.txt --gen_mode img2img
if [ $? -ne 0 ];
  then
    bug=`expr ${bug} + 1`;
fi

easydata --model ppdataaug --ori_data_dir demo/ocr_data/ --label_file demo/ocr_data/train_list.txt --gen_mode img2img --model_type ocr_rec
if [ $? -ne 0 ];
  then
    bug=`expr ${bug} + 1`;
fi

easydata --model ppdataaug --bg_img_dir demo/ocr_rec/bg --corpus_file demo/ocr_rec/corpus.txt --gen_mode text2img --model_type ocr_rec
if [ $? -ne 0 ];
  then
    bug=`expr ${bug} + 1`;
fi

easydata --model ppdataaug --ori_data_dir demo/shitu_data --label_file demo/shitu_data/train_list.txt --gen_mode img2img --use_big_model False
if [ $? -ne 0 ];
  then
    bug=`expr ${bug} + 1`;
fi

cp ./tests/test_PPDA.py ./
python3.7 test_PPDA.py
if [ $? -ne 0 ];
  then
    bug=`expr ${bug} + 1`;
fi

echo ${bug}
exit ${bug}
