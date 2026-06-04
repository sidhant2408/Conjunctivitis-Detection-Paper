# **incondet-v2**

All the scripts are written in Python version - 3.8.11.

===============================================================================

Project Organization
------------

    ├── README.md            <- The top-level README for project info.
    ├── Code
    │   ├── metadata         <- The intermediate loss, accuracy,f1 score values for future computations.
    │   ├── reports          <- The different loss and accuracy curves in different learning rates
    │   ├── model.py          
    │   ├── model0.py
    │   ├── train.py
    │   ├── trace_model.py
    │   ├── metascripts.py
    │   ├── visualize.py
    │   ├── EyeData_preparation.py
    │   ├── data_preprocess.py
    |
    |
    └── DATASET             <- Dataset to be used for model learning and testing
        ├── train           <- Training dataset
        |   ├── healthy           
        |   └── infected
        └── validation      <- Testing dataset
            ├── healthy           
            └── infected
        
        
    
    
## **Libraries required**

The following libraries are required to implement the project.


- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Torch
- TorchVision
- cv2
- PIL
- os


Getting Started
------------
1. Clone the repository
2. Open a terminal inside the directory  
3. Move DATASET(our prepared dataset) folder in the Code Folder  

## **Preparing Custom dataset**
1. Crop the healthy and infected images using the opensource tool [freehand-cropper](https://half-6.github.io/lf-freehand-cropper/)
2. Put the respective category images in two folders - healthy and infected and finally put them in datasets folder.
3. Run ./data_preprocess.py
4. Rename the datasets folder as DATASET and put inside the Code folder

## **Training and Testing**
Run ./train.py
## **Metric Analysis**
Run ./metascripts.py
## **Convert the model for app deployment**
Run ./trace_model.py



**App .apk file**
https://drive.google.com/drive/folders/1I4YH7zRPWfjzyoWZxiP-0mR4K9DrK9SM?usp=sharing

