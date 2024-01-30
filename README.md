## Framework
![img](https://github.com/Tntttt/LTFormer/blob/main/pic/framework.png)
Overview of the matching algorithm framework. The feature point detector is set to the default SIFT algorithm, while the feature descriptors are generated using LTFormer. These descriptors form a triplet to start self-supervised training to get the correspondence. Ultimately, this framework enables the matching of keypoints between visible light and near-infrared images.

## Dependencies
To simplify the reproduction steps, we only need to install
```shell script
conda env create -f environment.yml
conda activate LTFormer
```
## Dataset
1、Download from [SAM](https://github.com/AmberHen/WHU-OPT-SAR-dataset)
2、Strip visible and infrared channel from optical images.
3、Crop RGB and near-infrared images size of 926 × 926.

## Training
```shell script
python 
```
