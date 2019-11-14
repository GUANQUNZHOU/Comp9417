import cv2
import numpy as np
import pandas as pd
from glob import glob
import argparse
import pickle as pkl
from skimage import feature
from sklearn import metrics
# from sklearn.model_selection import train_test_split
import time, math
import os, pydicom
from collections import defaultdict
from mask_functions import *

# run: python3 SIIM-ACR_ML.py -i dicom-images-train -l dicom-images-train -m my_model.p -t dicom-images-test -o out

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--train_image_dir" , help="Path to train images", required=True)
    parser.add_argument("-l", "--label_dir", help="Path to labels", required=True)
    parser.add_argument("-m", "--model_path", help="Path to save model. End with .p", required = True)
    parser.add_argument("-t", "--test_image_dir" , help="Path to test images", required=True)
    parser.add_argument("-o", "--output_dir", help="Path to output directory", required = True)
    args = parser.parse_args()

    # validate the arguments
    if not os.path.exists(args.train_image_dir):
        raise ValueError("Train image directory does not exist", args.image_dir)
    if not os.path.exists(args.label_dir):
        raise ValueError("Label directory does not exist")
    if args.model_path.split('.')[-1] != "p":
        raise ValueError("Model should end with .p")
    if not os.path.exists(args.test_image_dir):
        raise ValueError("Test image directory does not exist")
    if not os.path.exists(args.output_dir):
        raise ValueError("Output directory does not exist")
    return args

def calc_lbp_features(img, neigh, radius):

    # print ('[INFO] Computing LBP features.')
    lbp_features = feature.local_binary_pattern(img, neigh, radius, method="default") # method="uniform")

    return np.array(lbp_features)

def create_patches(img, size):

    if img is None:
        print('######### NO IMAGE! ##########')
    pat = []
    for i in range(0, img.shape[0]//size):
        for j in range(0, img.shape[0]//size):
            if len(img.shape) == 2:
                pat.append(img[i*size:i*size+size, j*size:j*size+size])
            else:
                pat.append(img[i*size:i*size+size, j*size:j*size+size, :])

    return np.array(pat)
    
def create_features(img, label, train=True):

    radius = 12 # LBP radius
    neigh = radius * 8 # LBP number of neighbours
    num_examples = 120000 # number of examples per image to use for training model

    # print("image shape", img.shape)
    feature_img = np.zeros((img.shape[0],img.shape[1], 2))
    feature_img[:,:,0] = img

    lbp_features = calc_lbp_features(img, neigh, radius)
    feature_img[:,:,1] = lbp_features

    # make patches of size = neigh*2
    # size = neigh
    # feat_patches = create_patches(feature_img, size)
    # print('patch shape:', feat_patches.shape)

    features = feature_img.reshape((feature_img.shape[0]*feature_img.shape[1], feature_img.shape[2]))
    # features = feat_patches.reshape((feat_patches.shape[0]*feat_patches.shape[1]*feat_patches.shape[2], feat_patches.shape[3]))
    # print('patch reshape shape:', features.shape)

    if train == True:
        subsamples = np.random.randint(0, features.shape[0], num_examples)
        features = features[subsamples]
        # features = features.reshape((features.shape[0], features.shape[1]*features.shape[2], features.shape[3]))
        print('feature shape:', features.shape)
    else:
        subsamples = []
        # features = feat_patches.reshape((feat_patches.shape[0]* feat_patches.shape[1]*feat_patches.shape[2], feat_patches.shape[3]))
    
    # labels = label.reshape(label.shape[0]*label.shape[1], 1)
    if train == True:
        # label_patches = create_patches(label, size)
        labels = label.reshape(label.shape[0]*label.shape[1], 1)
        labels = labels[subsamples]
        print('label shape:', labels.shape)
    else:
        labels = None
        
    return features, labels

def create_training_dataset(image_list, label_list):

    print ('[INFO] Creating training dataset on %d image(s).' %len(image_list))
    X = []
    y = []

    for i, img in enumerate(image_list):
        print (f'[INFO] Computing LBP features for image {i+1}/{len(image_list)}.')
        features, labels = create_features(img, label_list[i])
        y.append(labels)
        X.append(features)
        
    X = np.array(X)
    X = X.reshape(X.shape[0]*X.shape[1], X.shape[2])
    y = np.array(y)
    y = y.reshape(y.shape[0]*y.shape[1], y.shape[2]).flatten()
    print('==========================')

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # return X_train, X_test, y_train, y_test
    return X, y

def train_model(X, y):

    classifier = 'rf'
    if classifier == 'svm':
        from sklearn.svm import SVC
        print ('[INFO] Training Support Vector Machine model.')
        model = SVC(gamma='auto', verbose=True)
        model.fit(X, y)
    elif classifier == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        print ('[INFO] Training Random Forest model.')
        model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42,  verbose=2, n_jobs=-1)
        model.fit(X, y)

    print ('[INFO] Model training complete.')
    print ('[INFO] Training Accuracy: %.2f' %model.score(X, y))
    return model

def compute_prediction(img, model):

    features, _ = create_features(img, label=None, train=False)
    predictions = model.predict(features.reshape(-1, features.shape[1]))
    pred_size = int(math.sqrt(features.shape[0]))
    predict_img = predictions.reshape(pred_size, pred_size)
    print('predicted image shape: %sx%s' %(predict_img.shape[0], predict_img.shape[1]))
    predict_img = np.multiply(predict_img.astype(np.uint8), 255)

    return predict_img

def main(need_train, on_windows, train_image_dir, label_dir, model_path, test_image_dir, output_dir):

    start = time.time()

    if need_train:
        ################################################################
        ############### Load the train images and labels ###############
        ################################################################

        train_files = sorted(glob(f'{train_image_dir}/*/*/*.dcm'))[:200]
        print(len(train_files), 'training files.')
        # load rle-encoded masks
        masks = pd.read_csv(f'{label_dir}/train-rle.csv')
        # images can have multiple annotations
        masks_ = defaultdict(list)
        for image_id, rle in zip(masks['ImageId'], masks[' EncodedPixels']):
            masks_[image_id].append(rle)
            
        annotated = {k: v for k, v in masks_.items() if v[0] != ' -1'}
        print("%d of %d images are annotated" % (len(annotated), len(masks_)))
        print()

        # # find indices of image contain diseases
        # have_disease = []
        # for i, fn in enumerate(train_files):
        #     if '-1' not in masks_[fn.split('\\')[-1][:-4]][0]:
        #         have_disease.append(i)
        # # only train the images with diseases
        # train_f = np.array(train_files)
        # train_files = train_f[have_disease[:-10]]        

        img_h = img_w = 1024
        train_images = np.zeros((len(train_files), img_h, img_w), dtype=np.uint8)
        train_labels = np.zeros((len(train_files), img_h, img_w), dtype=np.uint8)
        
        for i, fn in enumerate(train_files):
            cur_img = pydicom.read_file(fn).pixel_array
            # train_images[i] = np.expand_dims(cur_img, axis=2)
            train_images[i] = cur_img

            if on_windows:
                ## windows ##
                # print(fn.split('\\')[-1][:-4]) 
                if '-1' not in masks_[fn.split('\\')[-1][:-4]][0]:
                    tmp = rle2mask(masks_[fn.split('\\')[-1][:-4]][0], img_h, img_w)
                    # train_labels[i] = np.multiply(tmp, 255)
                    train_labels[i] = tmp
            else:
                ## unix/linux ##
                # print(fn.split('/')[-1])
                if '-1' not in masks_[fn.split('/')[-1][:-4]][0]:
                    tmp = rle2mask(masks_[fn.split('/')[-1][:-4]][0], img_h, img_w)  
                    train_labels[i] = tmp
            
        # resize the images to train images in non-overlapping patches
        # new_h = new_w = 256
        # train_images = train_images.reshape((-1, new_h, new_w, 1))
        # train_labels = train_labels.reshape((-1, new_h, new_w, 1))   

        ##################################################################
        ############### Train the model using train images ###############
        ##################################################################

        X_train, y_train = create_training_dataset(train_images, train_labels)
        model = train_model(X_train, y_train)
        pkl.dump(model, open(model_path, "wb"))
        print ('[INFO] Processing time:',time.time()-start)

    else:
        have_disease = range(0, 20)

    ###################################################
    ############# predict the test images #############
    ###################################################

    test_files = sorted(glob((f'{test_image_dir}/*/*/*.dcm')))[:20]  # dicom-images-test
    # test_files = train_f[have_disease[-20:]] 
    print ('[INFO] Predicting %s test images' %len(test_files))
    # model = pkl.load(open(model_path, "rb") )

    for file in test_files:
        print ('[INFO] Processing images:', os.path.basename(file))
        cur_img = pydicom.read_file(file).pixel_array
        predict_img = compute_prediction(cur_img, model)
        outfilename = os.path.basename(file).split('.')[:-1]
        cv2.imwrite(os.path.join(output_dir, '.'.join(outfilename))+'.png', predict_img)


if __name__ == "__main__":
    
    need_train = True
    on_windows = True

    # args = parse_args()
    # train_image_dir = args.train_image_dir
    # label_dir = args.label_dir
    # model_path = args.model_path

    # test_image_dir = args.test_image_dir
    # output_dir = args.output_dir
    train_image_dir = "dicom-images-train"
    label_dir = "dicom-images-train"
    model_path = "my_model.p"
    test_image_dir = "dicom-images-test"
    output_dir = "out"
    main(need_train, on_windows, train_image_dir, label_dir, model_path, test_image_dir, output_dir)
