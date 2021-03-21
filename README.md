# Colour Pop Effect
This code helps you to create the colour pop effect on any image. It works good in any image but we would advice to use when you have a good seperation between the person and the background. In other words it works well if the person is in center of the image. It woorks for only one person. You can see the system working in the below gif.
![alt text](https://github.com/Neha-Baranwal/colour_pop_effect/blob/main/colour_pop_effect.gif?raw=true)

## Working Principle
The working principle of the system is easy, we are using the object segmention in the backend. We used MASK-RCNN pretrained model for this purpose. Once you upload the image, the system runs the mask-rcnn model over that and then it reads if the person class exisit there and if yes it try to find the biggest person in the image. The biggest contour gives us the biggest person. The idea behind considering the biggest contour is bind with the idea to keep the person in center. If the person for whom you want to create the colour pop effect is in center, it would eventully become the bigget contour among all. 

### Installaion Help

You need to install following dependencis to run the code. You can use the requirements.txt to install these dependencies. To get the pretrained model, you can use the link below.Mask-RCNN Pretrained Model :- You may download the model from the below link.
https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5

```
* Tensorflow :- CPU/GPU 1.15
* Keras :- Version 2.31
* scikit-image
* ipython
* imgaug
* pycocotools
* h5py :- must be less than 3.0
```

## Running the tests

After running the code, you can access it though local host, http://127.0.0.1:5002 or by http://0.0.0.0:5002.


