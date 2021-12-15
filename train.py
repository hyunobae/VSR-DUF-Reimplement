# -*- coding:utf-8 -*-
""" 
@version: 01
@author:erichym
@license: Apache Licence 
@file: mytrain.py 
@time: 2018/12/08
@contact: yongminghe_eric@qq.com
@software: PyCharm
"""
import os
import tensorflow as tf
from utils import BatchNorm,Conv3D,DynFilter3D,depth_to_space_3D,Huber, LoadImage, CustomDataloader, Model
import numpy as np
import glob
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
# Size of input temporal radius
T_in = 7
# Upscaling factor
R = 4


def load_datasets(path):
    dir = os.listdir(path)
    dir.sort()
    frames = []
    for i in range(len(dir)):
        dir_frame = glob.glob(path+'/'+dir[i]+'/*.png')
        for f in dir_frame:
            frames.append(LoadImage(f))
            print(f"{f} appended")

    frames = np.asarray(frames)
    return frames


os.environ["CUDA_VISIBLE_DEVICES"]="3"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

"""
train datasets
"""
x_train_path='./data/testtrain/L/'
y_train_path='./data/testtrain/G/'
x_train_data=load_datasets(x_train_path) # print(x_data_padded.shape) (26, 100, 115, 3)
y_train_data=load_datasets(y_train_path) # print(y_data.shape) (20, 400, 460, 3)

x_train = x_train_data.reshape(x_train_data, [128, 7, -1, -1, 1])
y_train = y_train_data.reshape(y_train_data, [128, 7, -1, -1, 1])

"""
valid datasets
"""
x_valid_path='./data/val/L/'
y_valid_path='./data/val/G/'
x_valid_data=load_datasets(x_valid_path) # print(x_data_padded.shape) (26, 100, 115, 3)
y_valid_data=load_datasets(y_valid_path) # print(y_data.shape) (20, 400, 460, 3)


print(x_train_data.shape)
print(y_train_data.shape)


#valid_loader = Customdataloader(x_valid_padded, y_valid_data)
is_train = True

#cost=Huber(y_true=H_out_true,y_pred=out_H,delta=0.01)


# build model
model = Model()
model.summary()
modelpath = '/log/'+'{epoch:03d}--{val_acc:.4f}.pb'

checkpoint = ModelCheckpoint(modelpath, monitor='val_accuracy', verbose=0,
        save_best_only=True, mode='max')

def scheduler(epoch, lr):
    if epoch % 10 ==0:
        return lr * 0.1

adam = Adam(lr=0.001)

print(x_train_data.shape)

model.compile(optimizer=adam, loss=Huber, metrics=['accuracy'])
model.fit(x_train_data, y_train_data, epochs=200000, batch_size=128, 
        verbose=0, validation_data=(x_valid_data, y_valid_data), callbacks=[checkpoint, scheduler])


