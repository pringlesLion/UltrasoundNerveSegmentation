import numpy as np
np.random.seed(2016)

import os
import glob
import cv2
import pickle
import pandas as pd
import datetime
from shutil import copy2

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras import backend as K 
from keras.callbacks import EarlyStopping, ModelCheckpoint
from augment import ImageDataGenerator 

smooth = 1.
def np_dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return - dice_coef(y_true, y_pred)

def get_unet(img_rows, img_cols):
    inputs = Input((1, img_rows, img_cols))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', init='he_normal')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', init='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', init='he_normal')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', init='he_normal')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', init='he_normal')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', init='he_normal')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', init='he_normal')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', init='he_normal')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', init='he_normal')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', init='he_normal')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', init='he_normal')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', init='he_normal')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', init='he_normal')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)
    
    sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=1e-5)

    model.compile(optimizer=adam, loss=dice_coef_loss, metrics=[dice_coef])

    return model

def preprocess(imgs, img_rows, img_cols):
    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, 0] = cv2.resize(imgs[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
    return imgs_p

def prep(img):
    img = img.astype('float32')
    img = cv2.threshold(img, 0.5, 1., cv2.THRESH_BINARY)[1].astype(np.uint8)
    img = cv2.resize(img, (420, 580))
    return img

def read_and_normalize_train_data(img_rows, img_cols):
    imgs_train, imgs_mask_train, subject_id, unique_subjects = load_train_data(img_rows, img_cols)
    imgs_train = preprocess(imgs_train, img_rows, img_cols)
    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization
    
    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = preprocess(imgs_mask_train, img_rows, img_cols)
    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]
    print('Train shape:', imgs_train.shape)
    print('Train target shape:', imgs_mask_train.shape)
    print(imgs_train.shape[0], 'train samples')
    return imgs_train, imgs_mask_train, subject_id, unique_subjects

def read_and_normalize_test_data(img_rows, img_cols, mean, std):
    imgs_test, imgs_id_test = load_test_data(img_rows, img_cols)
    imgs_test = preprocess(imgs_test, img_rows, img_cols)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std
    print('Test shape:', imgs_test.shape)
    print(imgs_test.shape[0], 'test samples')
    return imgs_test, imgs_id_test

def get_im_cv2(path, img_rows, img_cols):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # Reduce size
    resized = cv2.resize(img, (img_cols, img_rows))
    return resized

def load_test_data(img_rows, img_cols):
    print('Read test images')
    test_data_path = os.path.join('test')
    images = os.listdir(test_data_path)
    total = len(images)
    imgs = np.ndarray((total, 1, img_rows, img_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype=np.int32)
    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    for image_name in images:
        img_id = int(image_name.split('.')[0])
        img = get_im_cv2(os.path.join(test_data_path, image_name), img_rows, img_cols)
        img = np.array([img])
        imgs[i] = img
        imgs_id[i] = img_id

        if i % 1000 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    return imgs, imgs_id

def load_train_data(img_rows, img_cols):
    print('Read train images')
    subject_id = []
    df = pd.read_csv('train_masks.csv')
    unique_subjects = df.subject.unique()
    total = df.shape[0]
    X_train = np.ndarray((total, 1, img_rows, img_cols), dtype=np.uint8)
    y_train = np.ndarray((total, 1, img_rows, img_cols), dtype=np.uint8)
    for i in range(total):
        image_name =  str(df['subject'][i]) + '_' + str(df['img'][i]) + '.tif'
        mask_name = str(df['subject'][i]) + '_' + str(df['img'][i]) + '_mask.tif'
        img = get_im_cv2(os.path.join('train', image_name), img_rows, img_cols)
        img = np.array([img])
        X_train[i] = img
        
        img_mask = get_im_cv2(os.path.join('train', mask_name), img_rows, img_cols)
        img_mask = np.array([img_mask])
        y_train[i] = img_mask
        subject_id.append(df['subject'][i])
    
    print(unique_subjects)
    print X_train.shape, y_train.shape, len(subject_id)
    return X_train, y_train, subject_id, unique_subjects
    
def copy_selected_subjects(train_data, train_target, subject_id, subject_list):
    data = []
    target = []
    index = []
    for i in range(len(subject_id)):
        if subject_id[i] in subject_list:
            data.append(train_data[i])
            target.append(train_target[i])
            #index.append(i)
    data = np.array(data)
    target = np.array(target)
    #index = np.array(index)
    return data, target#, index
	
def run_length_enc(label):
    from itertools import chain
    x = label.transpose().flatten()
    y = np.where(x > 0)[0]
    if len(y) < 10:  # consider as empty
        return ''
    z = np.where(np.diff(y) > 1)[0]
    start = np.insert(y[z+1], 0, y[0])
    end = np.append(y[z], y[-1])
    length = end - start
    res = [[s+1, l+1] for s, l in zip(list(start), list(length))]
    res = list(chain.from_iterable(res))
    return ' '.join([str(r) for r in res])

def run_single():
    # input image dimensions
    img_rows, img_cols = 112, 144
    batch_size = 32
    nb_epoch = 250
    random_state = 51

    train_data, train_target, subject_id, unique_subjects = read_and_normalize_train_data(img_rows, img_cols)
    #test_data, test_id = read_and_normalize_test_data(img_rows, img_cols)

    subject_list_train, subject_list_val = train_test_split(unique_subjects, test_size = 0.2, random_state=random_state)
    
    X_train, Y_train = copy_selected_subjects(train_data, train_target, subject_id, subject_list_train)
    X_valid, Y_valid = copy_selected_subjects(train_data, train_target, subject_id, subject_list_val)

    print('Start Single Run')
    print('Split train: ', len(X_train))
    print('Split valid: ', len(X_valid))
    print('Train subjects: ', subject_list_train)
    print('Valid subjects: ', subject_list_val)
   
    callbacks = [ModelCheckpoint('unet2.hdf5', monitor='val_loss', save_best_only=True)]
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        #width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
        #height_shift_range=0,  # randomly shift images vertically (fraction of total height)
        #shear_range=0.2,
        #zoom_range=0.2,
        #rescale=1.25,
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images
    model = get_unet(img_rows, img_cols)
    #model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, shuffle=True, verbose=1, 
    #           validation_data=(X_valid, Y_valid), callbacks=callbacks)
    datagen.fit(X_train)
    model.fit_generator(datagen.flow(X_train, Y_train,batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0]*2,
                        nb_epoch=nb_epoch,validation_data=(X_valid, Y_valid), callbacks=callbacks)
    model.load_weights('unet2.hdf5')
    predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)
    score = np_dice_coef(Y_valid, predictions_valid)
    print('Score dice coef: ', score)	
        
def run_cross_validation(nfolds=10):
    # input image dimensions
    img_rows, img_cols = 112, 144
    batch_size = 32
    nb_epoch = 50
    random_state = 51

    train_data, train_target, subject_id, unique_subjects = read_and_normalize_train_data(img_rows, img_cols)
    test_data, test_id = read_and_normalize_test_data(img_rows, img_cols)
    
    kf = KFold(len(unique_subjects), n_folds=nfolds, shuffle=True, random_state=random_state)
    num_fold = 0
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0,  # randomly shift images vertically (fraction of total height)
        shear_range=0.2,
        zoom_range=0.2,
        rescale=1.25,
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images
    for train_index, test_index in kf:
        model = get_unet(img_rows, img_cols)
        X_train, Y_train = copy_selected_subjects(train_data, train_target, subject_id, unique_subjects[train_index])
        X_valid, Y_valid = copy_selected_subjects(train_data, train_target, subject_id, unique_subjects[test_index])
        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))
        weights_name = 'unet_' + str(num_fold) + '.h5'
        callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=0), 
                     ModelCheckpoint(weights_name, monitor='val_loss', save_best_only=True)]  
        datagen.fit(X_train)
        model.fit_generator(datagen.flow(X_train, Y_train,batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch,validation_data=(X_valid, Y_valid), callbacks=callbacks)
        model.load_weights(weights_name)
        predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)
        score = np_dice_coef(Y_valid, predictions_valid)
        print('Score dice coef: ', score)

def submission(img_rows, img_cols, mean, std):
    imgs_test, imgs_id_test = read_and_normalize_test_data(img_rows, img_cols, mean, std)
    

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('unet.hdf5')

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test = model.predict(imgs_test, verbose=1)

    argsort = np.argsort(imgs_id_test)
    imgs_id_test = imgs_id_test[argsort]
    imgs_test = imgs_test[argsort]

    total = imgs_test.shape[0]
    ids = []
    rles = []
    for i in range(total):
        img = imgs_test[i, 0]
        img = prep(img)
        rle = run_length_enc(img)

        rles.append(rle)
        ids.append(imgs_id_test[i])

        if i % 1000 == 0:
            print('{}/{}'.format(i, total))

    first_row = 'img,pixels'
    suffix = str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join('submission_' + suffix + '.csv')

    with open(sub_file, 'w+') as f:
        f.write(first_row + '\n')
        for i in range(total):
            s = str(ids[i]) + ',' + rles[i]
            f.write(s + '\n')
		
if __name__ == '__main__':
    run_single()
    #run_cross_validation()
	
