import time
import threading
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
model = tf.keras.applications.mobilenet_v2.MobileNetV2()
resizedimg = image.img_to_array(img)
finalimg = np.expand_dims(resizedimg,axis=0)
finalimg = tf.keras.applications.mobilenet_v2.preprocess_input(finalimg)
#finalimg.shape

start_time = time.time()

for i in range(10):
    predictions = model.predict(finalimg)

end_time = time.time()
total_time = end_time - start_time


def run(model):
    model.predict(finalimg)


start = time.time()
for i in range(2):
    record_thread = threading.Thread(target=run, args=(model,))
    record_thread.start()
    record_thread.join()
end = time.time()
total_time = (end - start) * 1000



def summary_config():
    print("\n\n\n")
    print("----------------------- REPORT START ----------------------")
    print("Model name: {0}".format('mobilenet'))
    print("Num of samples: {0}".format(10))
    print("Thread num: {0}".format(2))
    print("Batch size: {0}".format(1))
    print("Device: {0}".format("GPU"))
    print(f"Average latency(ms): {total_time * 1000 / (2 * 10)}")
    print(f"QPS: {(10 * 1 * 2) / total_time}")
    print("------------------------ REPORT END -----------------------")
    print("\n")


summary_config()
