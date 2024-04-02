bash main.sh PaddleRec >log.rec 2>&1 
sleep 60
bash main.sh PaddleGAN >log.gan 2>&1 
sleep 60
bash main.sh PaddleOCR >log.ocr 2>&1 
sleep 60
bash main.sh PaddleSeg >log.seg 2>&1 
sleep 60
bash main.sh PaddleVideo >log.video 2>&1 
sleep 60
bash main.sh PaddleDetection >log.detection 2>&1 
sleep 60
bash main.sh PaddleNLP >log.nlp 2>&1 
sleep 60
bash main.sh PaddleClas >log.clas 2>&1 
