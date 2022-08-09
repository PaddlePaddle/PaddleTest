import time
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
#import matplotlib.pyplot as plt
from tensorflow.keras.applications import imagenet_utils
#from IPython.display import Image

# importing image
filename = './content/birds.jpg'

##  #displaying images
##  Image(filename,width=224,height=224)
img = image.load_img(filename,target_size=(224,224))
##  print(img)
##  plt.imshow(img)

#initializing the model to predict the image details using predefined models.
model = tf.keras.applications.resnet50.ResNet50()
resizedimg = image.img_to_array(img)
finalimg = np.expand_dims(resizedimg,axis=0)
finalimg = tf.keras.applications.resnet50.preprocess_input(finalimg)

start_time = time.time()

use_gpu = True
#use_gpu = False
if use_gpu:
    run_device = '/gpu:0'
else:
    run_device = '/cpu:0'

with tf.device(run_device):
    for i in range(500):
        predictions = model.predict(finalimg)

end_time = time.time()
total_time = end_time - start_time

print("latency_ms: ", total_time / 500 * 1000)
