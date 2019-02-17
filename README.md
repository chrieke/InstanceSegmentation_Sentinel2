## Deep Learning for Instance Segmentation of Agricultural Fields - Master thesis

![](figures/fieldsPlanet.jpg)
     
### Abstract  
This thesis aims to delineate agricultural field parcels from satellite images via deep learning 
instance segmentation. Manual delineation is accurate but time consuming, and many 
automated approaches with traditional image segmentation techniques struggle to capture 
the variety of possible field appearances. Deep learning has proven to be successful in 
various computer vision tasks, and might be a good candidate to enable accurate, 
performant and generalizable delineation of agricultural fields. Here, a fully convolutional 
instance segmentation architecture (adapted from Li et al., 2016), was trained on Sentinel-2 
image data and corresponding agricultural field polygons from Denmark. In contrast to many 
other approaches, the model operates on raw RGB images without significant pre- and 
post-processing. After training, the model proved successful in predicting field boundaries on 
held-out image chips. The results generalize across different field sizes, shapes and other 
properties, but show characteristic problems in some cases. In a second experiment, the 
model was trained to simultaneously predict the crop type of the field instance. Performance 
in this setting was significantly worse. Many fields were correctly delineated, but the wrong 
crop class was predicted. Overall, the results are promising and prove the validity of the deep 
learning approach. Also, the methodology offers many directions for future improvement.                    
     
![](figures/train_predict.jpg)    
    
![](figures/results.jpg)    
    
### Instructions
    
**1. FCIS & MXNet installation**   
Install FCIS and MXNet according to the instructions in the [FCIS repository](https://github.com/msracver/FCIS). The setup works well with an AWS EC2 P2 instance and the official AWS Deep Learning AMI (Ubuntu). Make sure that the installations were successfull by running the FCIS demo.   
	```
	python FCIS/fcis/demo.py
	```
   
**2. Data Preprocessing**    
Follow the instructions and run the code in the **preprocessing** notebook. This will prepare the Denmark LPIS field data and create the image chips and COCO format annotations. Place the preprocessed vector and image folders `.output/preprocessing/annotations` and `.output/preprocessing/images` in `.FCIS/data/coco`.
![](figures/preprocessing_demo.jpg)    
    
**3. Configuration preparation**    
Place the provided configuration file `.model/resnet_v1_101_coco_fcis_end2end_ohem.yaml` in `.FCIS/experiments/fcis/cfgs`. A more detailed description of the model and training parameters used for the thesis is given in chapter 3.3.    
Delete the annotations cache (neccessary every time you change a configuration parameter that could influence the model evaluation or training).    
	```
	rm -rf .FCIS/data/coco/annotations_cache/; rm -rf .FCIS/data/cache/COCOMask/  
	```
    
**4. Model prediction / evaluation**    
Run the prediction / model evaluation task via the model trained in the thesis. The resulting instance segmentation and object detection proposals will be saved to `denmark\cocoresult\val2014\results\detections_val2014_results.json`    
	```
	python experiments/fcis/fcis_end2end_test.py --cfg experiments/fcis/cfgs/resnet_v1_101_coco_fcis_end2end_ohem.yaml --ignore_cache
	```

**5. (Optional) Model training**
You can carry out your own model training with custom data or configurations.    
Delete existing model files:    
	```
	rm -rf /home/ubuntu/FCIS/output/fcis/coco/resnet_v1_101_coco_fcis_end2end_ohem/
	```
Run the training task:    
	```
	python experiments/fcis/fcis_end2end_train_test.py --cfg experiments/fcis/cfgs/resnet_v1_101_coco_fcis_end2end_ohem.yaml
	```

