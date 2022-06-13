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
model = tf.keras.applications.vgg16.VGG16()
resizedimg = image.img_to_array(img)
finalimg = np.expand_dims(resizedimg,axis=0)
finalimg = tf.keras.applications.vgg16.preprocess_input(finalimg)

predictions = model.predict(finalimg)
results = imagenet_utils.decode_predictions(predictions)
print(results)
