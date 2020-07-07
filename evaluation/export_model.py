from lipnet.lipreading.videos import Video
from lipnet.lipreading.visualization import show_video_subtitle
from lipnet.core.decoders import Decoder
from lipnet.lipreading.helpers import labels_to_text
from lipnet.utils.spell import Spell
from lipnet.model2 import LipNet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import numpy as np
import sys
import os
import tensorflowjs as tfjs
lipnet = LipNet(3,100,50,75,32,28)
lipnet.model.load_weights("C:/Projects/lipnet/evaluation/models/unseen-weights178.h5")
# lipnet.model.summary()
tfjs.converters.save_keras_model(lipnet.baseModel, "tfjsModelbase")