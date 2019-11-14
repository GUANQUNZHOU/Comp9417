Here is the link of the shared Google drive which contains our code sand datasets:
https://drive.google.com/drive/folders/15RthVabwL9pYWLoUjCCSmjUruJI8D3XA?usp=sharing


#################################
####### Machine Learning ########
#################################

Code filename: "SIIM-ACR_ML.py"

Our code for machine learning algorithm were developed and tested under Python 3.4.3 with following packages:
    - numpy 1.15.4
    - pandas
    - skimage
    - Scikit-learn
    - pydicom   -- package to read the DICOM format images
    - pickle, argparse

There are 5 arguments needed to run the code:
    -i: Path to train images, in .dcm format.
    -l: Path to label, which is the csv file in the same folder as train images
    -m: Path to save the model, with model name ending with '.p'
    -t: Path to test images
    -o: Path to output directory

e.g. python3 SIIM-ACR_ML.py -i dicom-images-train -l dicom-images-train -m my_model.p -t dicom-images-test -o out

There is an additional helper libray "mask_functions" to convert between RLE encoded data and mask images. 

NOTE:
If you are running the code on Windows OS, there is no further change. 
If running on Linux/Unix/MacOS, change the variable 'on_windows' to 'False' at bottom of the code (under [if __name__ == "__main__":])



##############################
####### Deep Learning ########
##############################

Code filename: "COMP9417_DeepLearning_Code.ipynb"

Our code for deep learning models were developed and tested on Kaggle Kernels using Keras.
Libariy Requirment:
1. tensorflow
2. keras
3. efficientnet [If running on Kaggle kernel(!pip install -U efficientnet==0.0.4)](when necessary)
4. albumentations [If running on Kaggle kernel(!pip install albumentations > /dev/null)](when necessary)
5. tqdm [If running on Kaggle kernel(!pip install tqdm)](when necessary)

Input Data set Requirments:
1. Given image data are in dcm format and they are a bit hard to deal with, so we write some code to transform to png format.

2. The code used for repacking images are written in "Image_repack.ipynb". Putting this file with folder "dicom-images-test" and "dicom-images-train" under the same path, 3 zip files (mask.zip, train.zip and test.zip) will be generated after running the code successfully.

3. Before running the code, uploading the zip file and upzip them or directly upload the unzipped folders onto Kaggle as a dataset.

4. Changing the corresponding paths in the hyper parameters section.

5. Ensuring the paths contains names: mask, train and test

Model selection:
We have successfully implemented two methods, namely U-net and U-net++
To select a specific model to run, simply uncomment the correspond code blocks and comment out the other one

Output:
As we are participating in an on-going Kaggle competition, the model outputs a submmission csv file.

NOTE:
The model takes time to train, usually a model trained on 20 epochs takes around 3 hours to finish.