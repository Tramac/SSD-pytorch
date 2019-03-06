# SSD: Single Shot MultiBox Object Detector

It is my first trying on object detection in Pytorch. This implement is mainly based on the [amdegroot's ssd.pytorch](https://github.com/amdegroot/ssd.pytorch).

## Datasets
 - Download VOC2007<br>
 [VOC2007 trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)<br>
 [VOC2007 test](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar)
 
 ## Training SSD
  - First download the fc-reduced [VGG-16](https://arxiv.org/abs/1409.1556) from [here](https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth), and put the file in the ```./weights``` dir
  - To train SSD need to change the parameter ```mode``` in ```main.py``` to 'train':
  ```
  python train.py
  ```
  
  ## Evaluation
  - Change the parameter ```mode``` in ```main.py``` to 'test':
  ```
  python train.py
  ```
  
  ## Run Demo
  - Down a pre-trained network
  - SSD300 trained on VOC0712(newst PyTorch weights): [https://s3.amazonaws.com/amdegroot-models/ssd300_mAP_77.43_v2.pth](https://s3.amazonaws.com/amdegroot-models/ssd300_mAP_77.43_v2.pth)
  ```
  python demo.py
  ```
  
  ## References
  - [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch)
