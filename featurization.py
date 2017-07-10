""" 
Loads 5MB version of SqueezeNet into Keras and removes top layers.
Variable `topNode` in function `top_layer` sets which layer to use as output.
Function featurize() can be called to export array of activations from image path.
"""

import numpy as np
from keras.applications.imagenet_utils import (
    decode_predictions, preprocess_input
)
from keras.preprocessing import image
import squeezenet


def top_layer(topNode=64):
    """ 
    Return SqueezeNet model with specified output layer
    Call `show_layers()` too see list, and change topNode to desired output layer
    """
    model = squeezenet.SqueezeNet()

    def show_layers():
        [print(i, m.name) for i, m in enumerate(model.layers)]

    n = 67 - topNode
    for i in range(1, n):

        model.layers.pop()
        model.outputs = [model.layers[-i].output]
        model.layers[-i].outbound_nodes = []

    print(model.layers[-1].name)
    return model


def featurize(image_path):
    """ Returns the weights at the node specified by `top_layer()`"""
    img = image.load_img(image_path, target_size=(227, 227))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    model = top_layer()
    preds = model.predict(x)
    return np.squeeze(preds)


def _variance_visualize(preds):
    """ Dev function to visualize weights """
    from PIL import Image as image
    from time import sleep

    experiment = preds.mean(axis=2)
    print(experiment.shape)
    im = image.fromarray(experiment, mode='L')
    im = im.resize((300, 300), image.LANCZOS)
    im.show()

    # preds = preds.var(axis=3) # Flatten arrays to 2d by getting variance of each pixel between class activations. Other visualization approaches include taking mean `preds.mean(axis=3)`, or taking a single layer. `preds.take(1, axis=2)`

    # For inline image dispay in ipynb or vscode jupyter extension:
    # im.save('test.jpg')
    # from IPython.core.display import Image, display
    # display(Image('test.jpg'))


_variance_visualize(featurize('../testimg/apbt.jpg'))
