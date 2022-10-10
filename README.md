# Noisy_MER
This repository contains the code for the paper 
```
Micro-expression recognition with noisy labels, Tuomas Varanka, Wei Peng, Guoying Zhao, IS&T Electronic
Imaging: Human Vision and Electronic Imaging 2021 (157-1-157-8)
```
link: https://doi.org/10.2352/ISSN.2470-1173.2021.11.HVEI-157

## Running the code
1. Setting up the datasets. Go to the datasets.py file and set the paths to the directories of the datasets and provide the excel file path containing the meta data.
2. pip install -r requirements.txt
3. Change the data paths from [datasets.py](datasets.py) for `casme2`, `smic`, and `samm`. Change both the `df_path`, which refers to the excel file containing metadata and the `dataset_path`, which refers to the cropped dataset root.
4. Run one of the main*.py files. For simplicity the four different files include the different methods with main.py including the baseline and remaining main*.py consisting of the methods proposed in the paper.

You can change CPU/GPU from the start of the code.\
This code uses a pre-computed optical flow using [1], but similar results can be achieved by using Farneback [2] and Dual TV L1 [3] optical flow from OpenCV.


![Mean of optical flows.](of_mean.PNG)
![Overview of noisy label technique.](noisy_mer.PNG)


[1]  D. Sun, S. Roth, and M. J. Black, “Secrets of optical flow estimation
and their principles,” in International Conference on Computer Vision
and Pattern Recognition (CVPR), 2010, pp. 2432–2439\
[2] https://docs.opencv.org/4.x/de/d9e/classcv_1_1FarnebackOpticalFlow.html \
[3] https://docs.opencv.org/3.4/dc/d47/classcv_1_1DualTVL1OpticalFlow.html
