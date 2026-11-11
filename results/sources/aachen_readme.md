# The Aachen Day-Night v1.1 Dataset 
This is an extended version of the original Aachen Day-Night dataset described in:
```
T. Sattler, W. Maddern, C. Toft, A. Torii, L. Hammarstrand, E. Stenborg, D. Safari, M. Okutomi, M. Pollefeys, J. Sivic, F. Kahl, T. Pajdla. 
Benchmarking 6DOF Outdoor Visual Localization in Changing Conditions. 
Conference on Computer Vision and Pattern Recognition (CVPR) 2018 
```
The extended version is described in this paper:
```
Z. Zhang, T. Sattler, D. Scaramuzza. 
Reference Pose Generation for Visual Localization via Learned Features and View Synthesis. 
arXiv:2005.05179, 2020 
```
Besides the original query images, the Aachen Day-Night v1.1 dataset contains an additional 
93 nighttime queries. In addition, it uses a larger 3D model containing additional images. 
These additional images were extracted from video sequences captured with different 
cameras.

For details on the original Aachen Day-Night dataset, please see the 
`README_Aachen-Day-Night.md` file.

In order to use the Aachen Day-Night v1.1 dataset, you need the original Aachen Day-Night 
dataset.

The  reference 6DOF poses for the query images will not be released. Rather, we provide an 
[evaluation service](http://visuallocalization.net/) (see below).


## License
The images of the Aachen Day-Night v1.1 dataset are licensed under a 
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/) and are intended for non-commercial academic use only. 
All other data also provided as part of the Aachen Day-Night v1.1 dataset, including the 3D model 
and the camera calibrations, is derived from these images. Consequently, all other data is 
also licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/) and intended for non-commercial academic use only.

We are not planning to make the dataset available for commercial use.

### Using the Aachen Day-Night v1.1 Dataset
By using the Aachen Day-Night v1.1 dataset, you agree to the license terms set out above.
If you are using the Aachen Day-Night v1.1 dataset, please cite **all** of the following papers:
```
@article{Zhang2020ARXIV,
author={Zhang, Zichao and Sattler, Torsten and Scaramuzza, Davide},
title={{Reference Pose Generation for Visual Localization via Learned Features and View Synthesis}},
journal = {arXiv},
volume  = {2005.05179},
year = {2020},
}

@inproceedings{Sattler2018CVPR,
  author={Sattler, Torsten and Maddern, Will and Toft, Carl and Torii, Akihiko and Hammarstrand, Lars and Stenborg, Erik and Safari, Daniel and Okutomi, Masatoshi and Pollefeys, Marc and Sivic, Josef and Kahl, Fredrik and Pajdla, Tomas},
  title={{Benchmarking 6DOF Outdoor Visual Localization in Changing Conditions}},
  booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2018},
}

@inproceedings{Sattler2012BMVC,
  author={Sattler, Torsten and Weyand, Tobias and Leibe, Bastian and Kobbelt, Leif},
  title={{Image Retrieval for Image-Based Localization Revisited}},
  booktitle = {British Machine Vision Conference (BMCV)},
  year = {2012},
}
```

### Privacy
We take privacy seriously. For this reason, we used software to automatically blur faces and 
license plates in the images before the release. We verified the results manually and added 
black bars for missed detections. If you have any concerns regarding the images and other 
data provided with this dataset, or find faces or license plates we have missed, please 
[contact us](mailto:torsat@chalmers.se).


## Provided Files
The following files are provides with this release of the Aachen Day-Night v1.1 dataset in 
various directories:
* `3D-models/`: Contains the 3D model created from the reference images in various formats.
* `images_upright/`: Contains the images of the Aachen Day-Night dataset.
* `query/`: Contains the intrinsic calibrations of all query images used in the dataset.

All files are provided in a single zip file `aachen_v1_1.zip`. It should be placed in the same 
directory as the original Aachen Day-Night dataset and extracted there.
 
In the following, we will describe the different files provided with the dataset in more detail.

### 3D Model
This directory contains the extended reference 3D model build for the dataset. 
We provide the 3D reconstruction in COLMAP's [binary format](https://colmap.github.io/format.html#binary-file-format). In addition, we also provide a version in the 
[NVM file format](http://ccwu.me/vsfm/doc.html#nvm) used by VisualSfM [3,4]. 
The models use the same coordinate system as the  `aachen_cvpr2018_db.nvm` model 
provided by the original dataset. All images shared with `aachen_cvpr2018_db.nvm` should have 
the same camera pose.


### Images
The `images_upright/` directory contains the additional query images and the sequences 
used for the extended 3D model. The content of the directory should be places in the original 
`images_upright/` folder provided by the original Aachen Day-Night dataset. 

### Queries
We provide a text files with information about the intrinsics of all nighttime query images, both 
those of the original Aachen Day-Night dataset and the new nighttime queries.  The text file 
has the following file format: 
`image_name SIMPLE_RADIAL w h f cx cy r`
Here, `image_name` is the name of the query image. The name includes the subfolder names. 
The path of the image is relative to the `images_upright/` directory. The remainder of the 
line describes the intrinsic calibration of the camera. We use the `SIMPLE_RADIAL` model used 
in COLMAP (see [here](https://colmap.github.io/cameras.html) for details). 
The intrinsic calibration is defined by the width `w` and height `h` of the image, its focal length 
`f`, the position of the principal point (`cx` and `cy`), and a radial distortion parameter `r`. 


## Evaluating Results on the Aachen Day-Night v1.1 Dataset
We have set up an [evaluation service](http://visuallocalization.net/) for the dataset. You are 
able to upload the poses estimated by your method to this service, which in turn will provide 
results to you. 

Please submit your results as a text file using the following file format. For each query 
image for which your method has estimated a pose, use a single line. This line should store the
result as `name.jpg qw qx qy qz tx ty tz`.  
Here,  `name` corresponds to the filename of the image, without any directory names. 
`qw qx qy qz` represents the **rotation** from world to camera coordinates as a 
**unit quaternion**. `tx ty tz` is the camera **translation** (**not the camera position**). 
An example, obtained using the *DenseVLAD* baseline from the CVPR 2018 paper, for such a 
line is 
```
IMG_20140520_182846.jpg 0.004300400000000 -0.009711270000000 0.087104600000000 0.996143000000000 724.380999999999972 23.511800000000008 180.934000000000026
```
Note that the type of condition (day or night) **is not specified** as there is a unique mapping 
from image names to conditions.

Please adhere to the following naming convention for the files that you submit:
```
Aachen_v1_1_eval_[yourmethodname].txt
```
Here, `yourmethodname` is some name or identifier chosen by yourself. This name or identifier 
should be as unique as possible to avoid confusion with other methods. Once the evaluation 
service is ready, it will be used to display the results of your method.

**IMPORTANT:** Our evaluation tools expect that the coordinate system in which the camera 
pose is expressed is the **COLMAP coordinate system** (which is the same as the coordinate 
system of the NVM file). For details, please see the README file of the original Aachen 
Day-Night dataset. 

**IMPORTANT:** When submitting to the Aachen Day-Night v1.1 benchmark, make sure to 
submit results for **all** query images, i.e., the queries provided by the original Aachen 
Day-Night dataset and the queries provided by version 1.1.

## Acknowledgements
Some of the database images used to build the 3D model for the Aachen Day-Night dataset 
were recorded by Martin Habbecke, Ming Li, Robert Menzel, Torsten Sattler, Dominik Sibbing, 
Patrick Sudowe, and Tobias Weyand.

[Mapillary](https://www.mapillary.com/) kindly blurred the images of the Aachen Day-Night 
dataset to preserve privacy.

## References:
1. J. L. Schönberger, J.-M. Frahm. Structure-from-Motion Revisited. Conference on
Computer Vision and Pattern Recognition (CVPR) 2016.
2. J. L. Schönberger, E. Zheng, M. Pollefeys, J.-M. Frahm. Pixelwise View Selection for
Unstructured Multi-View Stereo. European Conference on Computer Vision (ECCV) 2016.
3. C. Wu. Towards Linear-time Incremental Structure From Motion. 3DV 2013
4. C. Wu. VisualSFM: A Visual Structure from Motion System. http://ccwu.me/vsfm/, 2011
