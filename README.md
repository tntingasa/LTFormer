## Abstract
Matching visible and near-infrared (NIR) images remains a significant challenge in remote sensing image fusion. The substantial nonlinear radiometric differences between heterogeneous images make the image matching task even more difficult. In recent years, deep learning has gained substantial attention in computer vision tasks, but many methods rely on supervised learning and necessitate large amounts of annotated data. However, annotated data is often scarce in the field of remote sensing image matching. To address this issue, this paper proposes a novel keypoint descriptor method that obtains more reliable feature descriptors through a self-supervised matching network. A Light-weight Transformer network called LTFormer is designed to generate deep feature descriptors. Additionally, a triplet loss function, LT Loss, is introduced to further enhance the matching performance. Compared to standard hand-crafted local feature descriptors, our method performs well in terms of performance. When compared to state-of-the-art deep learning-based methods, our approach is equally competitive while allowing training in the absence of annotated data.
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
1、Download from [WHU-OPT-SAR](https://github.com/AmberHen/WHU-OPT-SAR-dataset)  
2、Strip visible and infrared channel from optical images.  
3、Crop RGB and near-infrared images size of 926 × 926.  
4、Generation of triplet datasets  
```shell script
python run_generate_dataset.py
```
## Training
```shell script
python run_training_ltformer.py
```
## Testing and visualisation
The weights have been uploaded to the folder model/.
```shell script
python run_matching_demo.py
```
## Comparison to state-of-the-art handcrafted descriptors
![img](https://github.com/Tntttt/LTFormer/blob/main/pic/compare.png)
## Comparison to state-of-the-art learning feature descriptors
![img](https://github.com/Tntttt/LTFormer/blob/main/pic/compare_2.png)

