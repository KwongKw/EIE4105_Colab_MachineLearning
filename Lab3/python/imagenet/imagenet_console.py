from keras.applications.resnet import ResNet50
from keras.preprocessing import image
from keras.applications.resnet import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet')

img_path = [ 
'images/brown_bear.jpg'     ,'images/polar_bear.jpg'      ,'images/black_bear.jpg'      ,'images/2044930246_1053660e05.jpg' ,
'images/n07697100_10590.jpg','images/n03581125_91614.jpg' ,'images/n03581125_50853.jpg' ,'images/n02880940_4498.jpg'        ,
'images/n07768694_751.jpg'  ,'images/n07768694_354.jpg'   ,'images/n02123045_2274.jpg'  ,'images/n02123045_2033.jpg'   
]


for i in img_path:
  img = image.load_img(i, target_size=(224, 224))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)

  preds = model.predict(x)
  # decode the results into a list of tuples (class, description, probability)
  # (one such list for each sample in the batch)
  print('Predicted [',i[7:].strip(),'] :', decode_predictions(preds, top=3)[0])
