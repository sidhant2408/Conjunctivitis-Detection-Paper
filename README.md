## 👁️ A Machine Learning-based Pipeline to Detect Conjunctivitis from Images Taken with Mobile Camera
<p>
<img src="https://img.shields.io/badge/Python-3.8.11-blue">
<img src="https://img.shields.io/badge/PyTorch-Deep_Learning-red">
<img src="https://img.shields.io/badge/OpenCV-Computer_Vision-green">
<img src="https://img.shields.io/badge/Android-App-success">
</p>

<h3>Overview</h3>
<p>
Pink eye (conjunctivitis) is a common and contagious eye infection that can be caused by bacteria or viruses, and early detection is important to prevent spreading it to others and to spot possible links to illnesses like COVID-19. This study explored using a mobile app powered by artificial intelligence to screen for conjunctivitis by analyzing eye images. After testing seven different AI models, researchers selected the best-performing one, which achieved 98.4% accuracy in identifying whether someone has the condition. In real-world terms, this means people could potentially use their smartphones for quick, early screening, helping them seek treatment sooner and reduce the risk of infecting others. An complete pipeline can be visualized as per the below image.</p>

<img width="1036" height="624" alt="ecb36077-48e6-48f6-812b-32ad5ca41c55" src="https://github.com/user-attachments/assets/ccf3541c-6ceb-4fd9-bf8c-af682ed93597" />
<br><i>* This is an AI generated image. It depicts the actual flow of the complete project.</i>

## **Libraries required**

<img width="296" height="221" alt="image" src="https://github.com/user-attachments/assets/82fb5a25-4f24-4f13-a568-04ccfb20b8dc" />
<br><i>* This is an AI generated image</i>

## Model Pipeline Flow
<img width="677" height="396" alt="image" src="https://github.com/user-attachments/assets/8b9b273b-530e-4542-9e13-6ca0a3312b63" />
<br><i>* This is an AI generated image. It depicts the actual flow of the complete model.</i>


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

