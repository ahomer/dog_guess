from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
from keras import backend as K

from dog_names import *
from tqdm import tqdm
import numpy as np


def path_to_tensor(img_path):
    '''
    Convert image to tensor.

    INPUT:
        img_paths - the file path of images
    OUTPUT:
        list of image tensor by (1, 224, 224, 3)
    '''
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths):
    '''
    Convert image to tensor.

    INPUT:
        img_paths - the folder of images
    OUTPUT:
        list of image tensor by [row,with,height,channel]
    '''
    list_of_tensors = [path_to_tensor(img_path)
                       for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


def extract_VGG19(tensor):
    '''
    Convert tensor to VGG19 input 

    INPUT:
        tensor - the tensor of image
    OUTPUT:
        bottleneck feature of VGG19
    '''
    from keras.applications.vgg19 import VGG19, preprocess_input
    return VGG19(weights='imagenet', include_top=False).predict(preprocess_input(tensor))


def load_VGG19(model_file):
    '''
    Loading VGG19 pre-trained model

    INPUT:
        model_file - pre-trained model file
    OUTPUT:
        VGG19_model
    '''
    VGG19_model = Sequential()
    VGG19_model.add(GlobalAveragePooling2D(input_shape=(7, 7, 512)))
    VGG19_model.add(Dense(256, activation='relu'))
    VGG19_model.add(Dropout(0.5))
    VGG19_model.add(Dense(133, activation='softmax'))

    VGG19_model.load_weights(model_file)
    return VGG19_model


def VGG19_predict_breed(VGG19_model, img_path):
    '''
    Predict dog breed by VGG19 model

    INPUT:
        VGG19_model - pre-trained model file
        img_path - image file of one dog
    OUTPUT:
        dog breed
    '''
    # extract bottleneck features
    bottleneck_feature = extract_VGG19(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = VGG19_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]


def find_dog(img_path):
    K.clear_session()

    model = load_VGG19('../models/weights.best.VGG19.hdf5')
    breed = VGG19_predict_breed(model, img_path)
    breed = breed.rsplit('.', 1)[-1].strip()
    breed = "Ha ha,I think the dog's name is <br> <b>" + breed + "<b>"
    return breed


if __name__ == "__main__":
    model_file = '../models/weights.best.VGG19.hdf5'
    VGG19_model = load_VGG19(model_file)

    print('\nDog guess start...')
    img1 = '../data/dog_images/test/133.Yorkshire_terrier/Yorkshire_terrier_08346.jpg'
    print('\nprocessing ', img1)
    print(VGG19_predict_breed(VGG19_model, img1))

    img2 = '../data/dog_images/test/132.Xoloitzcuintli/Xoloitzcuintli_08312.jpg'
    print('\nprocessing ', img2)
    print(VGG19_predict_breed(VGG19_model, img2))

    img2 = '../data/uploads/Brittany_02625.jpg'
    print('\nprocessing ', img2)
    print(VGG19_predict_breed(VGG19_model, img2))
