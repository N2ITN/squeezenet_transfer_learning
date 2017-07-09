import numpy as np
import squeezenet
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image
import cv2
img = cv2.imread('testimg/roxy.jpg')

model = squeezenet.SqueezeNet(input_shape=img.shape)

img = image.load_img('testimg/roxy.jpg')
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)


def return_array():
    return preds.shape


def show_results():
    labels = sorted({p[1]: p[2] for p in decode_predictions(preds)[0]}.items(),
                    key=lambda x: x[1],
                    reverse=True)
    print()
    [print(z[0] + ': ' + str(round(z[1] * 100, 2)) + '%') for z in labels]
    print()


show_results()