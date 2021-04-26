############################################# Imports ###############################################################
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# from coco import coco
from mrcnn.config import Config
import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
config1 = tf.ConfigProto()
config1.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config1))
######################################################################################################################
# Root directory of the project
ROOT_DIR=os.getcwd()
print("Root Path|",ROOT_DIR)
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR,"logs", "sitizen20210404T2303","mask_rcnn_sitizen_0160.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "datasets", "Holoturian")
######################################################################################################################
class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "sitizen"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 5

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # currently training for only one class "Holoturian"
######################################################################################################################    
class InferenceConfig(CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.88

config = InferenceConfig()
config.display()
######################################################################################################################
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)
print('-------------------------- Weights are loaded successfully!--------------------------------')
######################################################################################################################
# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'holoturian']
######################################################################################################################
# ## Run Object Detection
# Load a random image from the images folder
#file_names = next(os.walk(IMAGE_DIR))[2]
imgae_name=os.path.join(IMAGE_DIR, "session_2017_11_04_kite_Le_Morne_G0191707.JPG")
print("Image|",imgae_name)
image = skimage.io.imread(imgae_name)

# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
score=list(r['scores'])
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'])
print("...................................................................................................")
print('clas ids:',r['class_ids'])
print('scores:',r['scores'])
print('bbox:',r['rois'])
selected_class_names=[class_names[class_id] for class_id in r['class_ids']]
print(selected_class_names)
######################################################################################################################




