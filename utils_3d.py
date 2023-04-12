################################################################################################
import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy.io import loadmat
from time import time

from skimage.metrics import mean_squared_error as img_mse
from skimage.metrics import structural_similarity

import keras.backend as K
from keras import Model, Input
from tensorflow_addons.layers import InstanceNormalization, GELU
from keras.layers import BatchNormalization, LayerNormalization, PReLU
from keras.layers import Flatten, Reshape, Concatenate, Lambda
from keras.layers import SeparableConv2D, AveragePooling2D, UpSampling2D, Dense
from keras.optimizers import Adam
from keras.losses import mean_squared_error as loss_mse
################################################################################################

def check_tensorflow_gpu():
    sys_info = tf.sysconfig.get_build_info()
    print('Tensorflow built with CUDA?',  tf.test.is_built_with_cuda())
    print('Tensorflow version:', tf.__version__)
    print('# GPU available:', len(tf.config.experimental.list_physical_devices('GPU')))
    print("CUDA: {} | cuDNN: {}".format(sys_info["cuda_version"], sys_info["cudnn_version"]))
    print(tf.config.list_physical_devices())
    return None