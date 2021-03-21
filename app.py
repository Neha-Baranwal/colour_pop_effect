from flask import Flask, request, Response,send_file,render_template,send_from_directory
import os
import traceback
import cv2 as cv
import numpy as np

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
from coco import coco

import tensorflow as tf
# to limit the gpu usage
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
# set_session(tf.Session(config=config))

###########################################################################################################################
# Initialize the Flask Application
app = Flask(__name__)
###########################################################################################################################
UPLOAD_FOLDER = os.path.basename('/uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
###########################################################################################################################
# Root directory of the project
# Directory to save logs and trained model
MODEL_DIR = os.path.join(os.getcwd(),"logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(os.getcwd(),"models","mask_rcnn_coco.h5")
print(COCO_MODEL_PATH)
######################################################################################################################
# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
######################################################################################################################
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

######################################################################################################################
# Create model object in inference mode.
# model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# # Load weights trained on MS-COCO
# model.load_weights(COCO_MODEL_PATH, by_name=True)
# model.keras_model._make_predict_function()
###########################################################################################################################
@app.route("/")
def hello():
    return render_template('index.html')
    #return "Flask Server is On..."
###########################################################################################################################
@app.route('/processImage', methods=['POST'])
def afterUpload():
    try:
        # Create model object in inference mode.
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

        # Load weights trained on MS-COCO
        model.load_weights(COCO_MODEL_PATH, by_name=True)
        model.keras_model._make_predict_function()
        
        print(".......................................................................................................")
        file = request.files['image']
        fileName = os.path.join("static",app.config['UPLOAD_FOLDER'],file.filename)
        file.save(fileName)
        #print("File|",file.filename)
        #print("Current Directory|",os.getcwd())
        print("Location|",fileName)
        image=cv.imread(fileName,1)
        print("Pass-0",image.shape)
        # print('Model Name:',model)
        try:
            print("Going to execute the model...")
            results = model.detect([image], verbose=1)
            print('Model executed...done!')
        except:
            #a=os.path.join(os.getcwd(),fileName)
            #command='python ColourPop.py "'+a+'"'
            #print(command)
            #os.system(command)
            print("Error...",traceback.print_exc())
        print("Pass-1")
        # Visualize results
        r = results[0]
        # print(r)
        rois=r['rois'];class_ids=r['class_ids'];scores=r['scores'];masks=r['masks']
        print("Results Retrieved!")
        area=[(item[3]-item[1])*(item[2]-item[0]) for item in r['rois']]
        print("Area|",area)
        biggestBox=area.index(max(area))
        print("Biggest Box|",biggestBox)
        print(biggestBox,r['rois'][biggestBox])
        print("-------------------------------------------------------------")
        masked_image=image.copy()
        mask = r['masks'][:, :, biggestBox]
        selected_class_id=class_ids[biggestBox]
        selected_person_name=class_names[selected_class_id]
        print('Class Name:',selected_person_name)
        print("Mask Created!")
        for c in range(3):
            masked_image[:, :, c] = np.where(mask != 1,masked_image[:, :, 0] ,masked_image[:, :, c])
        print("Image Created!")
        # current_directory=os.getcwd()
        # print('Current Directory:',current_directory)
        outputImageName="static/uploads/"+(file.filename).split(".")[0]+"_Pop."+(file.filename).split(".")[1]
        print(outputImageName)
        cv.imwrite(outputImageName,masked_image)
        outputfileName = os.path.join("static",app.config['UPLOAD_FOLDER'],(file.filename).split(".")[0]+"_Pop."+(file.filename).split(".")[1])
        print(outputfileName)
        #########################################################################################################
        return render_template('ColourPopEffect.html',src_image=fileName,dst_image=outputfileName,error="", init=True)
    except:
        return render_template('ColourPopEffect.html',src_image=fileName,dst_image=fileName,error="YES", init=True)
###########################################################################################################################
###########################################################################################################################
if __name__ == "__main__":
    app.debug=True
    app.run(host="0.0.0.0",use_reloader=True, port=5002,threaded=True)
