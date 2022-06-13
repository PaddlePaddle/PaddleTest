import time
import argparse
import threading
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
#import matplotlib.pyplot as plt
from tensorflow.keras.applications import imagenet_utils
#from IPython.display import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--repeat_num", type=int)
    parser.add_argument("--thread_num", type=int)
    parser.add_argument("--use_gpu")
    args = parser.parse_args()
    return args


args = parse_args()



# importing image
filename = './content/birds.jpg'

##  #displaying images
##  Image(filename,width=224,height=224)
img = image.load_img(filename,target_size=(224,224))
##  print(img)
##  plt.imshow(img)

#initializing the model to predict the image details using predefined models.
if args.model == 'vgg16':
    model = tf.keras.applications.vgg16.VGG16()
elif args.model == 'resnet50':
    model = tf.keras.applications.resnet50.ResNet50()

resizedimg = image.img_to_array(img)
finalimg = np.expand_dims(resizedimg,axis=0)
if args.model == 'vgg16':
    finalimg = tf.keras.applications.vgg16.preprocess_input(finalimg)
elif args.model == 'resnet50':
    finalimg = tf.keras.applications.resnet50.preprocess_input(finalimg)
#finalimg.shape


if args.use_gpu:
    run_device = '/gpu:0'
else:
    run_device = '/cpu:0'

def run(model):
    with tf.device(run_device):
        for i in range(args.repeat_num):
            predictions = model.predict(finalimg)


start = time.time()
for i in range(args.thread_num):
    record_thread = threading.Thread(target=run, args=(model,))
    record_thread.start()
    record_thread.join()
end = time.time()
total_time = end - start



def summary_config():
    print("\n\n\n")
    print("----------------------- REPORT START ----------------------")
    print("Model name: {0}".format(args.model))
    print("Num of samples: {0}".format(args.repeat_num))
    print("Thread num: {0}".format(args.thread_num))
    print("Batch size: {0}".format(1))
    print("Device: {0}".format("GPU" if args.use_gpu else "CPU"))
    print(f"Average latency(ms): {total_time * 1000 / (args.thread_num * args.repeat_num)}")
    print(f"QPS: {(args.repeat_num * 1 * args.thread_num) / total_time}")
    print("------------------------ REPORT END -----------------------")
    print("\n")


summary_config()
