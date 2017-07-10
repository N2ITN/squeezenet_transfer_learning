""" Send images, with directory as label, through featurizer, returning """
from glob import glob
from random import shuffle
import numpy as np
from featurization import featurize

all_inputs = glob('../*morning*/*.png')


def extract_features(all_inputs):
    """ Yields (label, feature) tuples. 
    Images labeled by directory name, featurized by featurization.py. """
    shuffle(all_inputs)
    for i, png in enumerate(all_inputs):
        yield png.split('/')[-2], featurize(png)


print(next(extract_features(all_inputs)))
