###################################################
#
#   Script to:
#   - Load the images and extract the patches
#   - Define the neural network
#   - perform the training
#
##################################################


import numpy as np
import configparser as ConfigParser

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.utils.vis_utils import plot_model as plot
from keras.optimizers import SGD
from keras.models import model_from_json
import sys
sys.path.insert(0, './lib/')
#sys.path.insert(0, '/Users/elamon/PycharmProjects/cells/retina-unet/')

#function to obtain data for training
from extract_patches import get_data_training

from help_functions import *


#Define the neural network. This will be used for training it from the scratch
def get_unet(n_ch,patch_height,patch_width):
    inputs = Input(shape=(n_ch,patch_height,patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv3)

    up1 = UpSampling2D(size=(2, 2))(conv3)
    up1 = concatenate([conv2,up1],axis=1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv4)
    #
    up2 = UpSampling2D(size=(2, 2))(conv4)
    up2 = concatenate([conv1,up2], axis=1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv5)
    #
    conv6 = Conv2D(2, (1, 1), activation='relu',padding='same',data_format='channels_first')(conv5)
    conv6 = core.Reshape((2,patch_height*patch_width))(conv6)
    conv6 = core.Permute((2,1))(conv6)
    ############
    conv7 = core.Activation('softmax')(conv6)

    model = Model(inputs=inputs, outputs=conv7)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    #if optimizer= 'sgd', the default values for the hyperparameters will be used.
    model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])

    return model



#========= Loading the parameters from the config file
config = ConfigParser.RawConfigParser()
config.read('./configuration.txt')

#path to the dataset
path_data = config.get('data paths', 'path_local')
#Name of the experiment (This is where the weights and the predictions will be saved)
name_experiment = config.get('experiment name', 'name')
#training settings
N_epochs = int(config.get('training settings', 'N_epochs'))
batch_size = int(config.get('training settings', 'batch_size'))
fine_tuning = config.get('training settings', 'finetuning')


path_experiment = './' +name_experiment +'/'
best_last = config.get('testing settings', 'best_last')

print (path_data)
print ("about to enter get_data_training")
#============ This function extracts patches from the training samples
patches_imgs_train, patches_masks_train = get_data_training(
    DRIVE_train_imgs_original = path_data + config.get('data paths', 'train_imgs_original'),
    DRIVE_train_groudTruth = path_data + config.get('data paths', 'train_groundTruth'),
    DRIVE_train_bordermasks = path_data + config.get('data paths', 'train_border_masks'),
    patch_height = int(config.get('data attributes', 'patch_height')),
    patch_width = int(config.get('data attributes', 'patch_width')),
    N_subimgs = int(config.get('training settings', 'N_subimgs')),
    inside_FOV = config.getboolean('training settings', 'inside_FOV') #select the patches only inside the FOV  (default == True)
)


#========= Save a sample of what you're feeding to the neural network ==========
N_sample = min(patches_imgs_train.shape[0],40)
visualize(group_images(patches_imgs_train[0:N_sample,:,:,:],5),'./'+name_experiment+'/'+"sample_input_imgs")#.show()
visualize(group_images(patches_masks_train[0:N_sample,:,:,:],5),'./'+name_experiment+'/'+"sample_input_masks")#.show()


#=========== Construct and save the model architecture =====
n_ch = patches_imgs_train.shape[1]
patch_height = patches_imgs_train.shape[2]
patch_width = patches_imgs_train.shape[3]


if fine_tuning=="False":
    print ("entered unet")
    model = get_unet(n_ch, patch_height, patch_width)  #the U-net model
    plot(model, to_file='./'+name_experiment+'/'+name_experiment + '_model.png')   #check how the model looks like
    json_string = model.to_json()
    open('./'+name_experiment+'/'+name_experiment +'_architecture.json', 'w').write(json_string)

elif fine_tuning=="True":
    # --------------------Fine tuning while training -----------------
    print ("About to load the pre trained model for fine tuning")
    model = model_from_json(open(path_experiment+name_experiment +'_architecture.json').read())
    model.load_weights(path_experiment+name_experiment + '_'+best_last+'_weights.h5')
    print ("loaded the pre trained model")

    print (model.layers[:8])
    print (model.layers[17:25])

    for layer in model.layers[:8]:
        layer.trainable=False
        print (layer)
        print ("these layers wont be trained")
    for layer in model.layers[17:25]:
        layer.trainable = False
        print (layer)
        print ("these layers wont be trained")
    model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])
#print model.layers
print ("Check: final output of the network:")
print (model.output_shape)




#============  Training ==================================
checkpointer = ModelCheckpoint(filepath='./'+name_experiment+'/'+name_experiment +'_best_weights.h5', verbose=1, monitor='val_loss', mode='auto', save_best_only=True)



patches_masks_train = masks_Unet(patches_masks_train)  #reduce memory consumption
model.fit(patches_imgs_train, patches_masks_train, nb_epoch=N_epochs, batch_size=batch_size, verbose=2, shuffle=True, validation_split=0.1, callbacks=[checkpointer])


#========== Save and test the last model ===================
model.save_weights('./'+name_experiment+'/'+name_experiment +'_last_weights.h5', overwrite=True)



