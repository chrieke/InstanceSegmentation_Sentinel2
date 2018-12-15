# InstanceSegmentation_Sentinel2  

**Deep Learning for Instance Segmentation of Agricultural Fields - Master thesis**  
     
![](figures/fieldsPlanet.jpg)
     
**Abstract**  
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
     
     
![](figures/results.jpg)    