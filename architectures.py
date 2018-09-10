'''
All model architectures used in the paper titled:
    "Multimodal Representation of Advertisements UsingSegment-level Autoencoders"
    published at ICMI 2018
'''


import os, json, glob
import numpy as np
import keras
from keras.layers import *
from keras.models import Model
from keras import metrics
import pandas as pd
import tensorflow as tf
from keras import callbacks
from keras.regularizers import l2

def v2v_512():
    v_input = Input(shape = (5632,), name='video_input')
    #v_input_noise = GaussianNoise(0.2)(v_input)
    v_1 = Dense(5632//2, activation='relu')(v_input)
    v_1_dropout = Dropout(0.2)(v_1)
    v_2 = Dense(5632//4, activation='relu')(v_1_dropout)
    v_3 = Dense(512, activation="relu")(v_2)
    v_4 = Dense(5632//4, activation='relu')(v_3)
    v_4_dropout = Dropout(0.2)(v_4)
    v_5 = Dense(5632//2, activation='relu')(v_4_dropout)
    v_6 = Dense(5632, activation='relu')(v_5)

    model = Model(inputs=v_input, outputs=v_6)
    return model

def v2v_256():
    v_input = Input(shape = (5632,), name='video_input')
    #v_input_noise = GaussianNoise(0.2)(v_input)
    v_1 = Dense(5632//2, activation='relu')(v_input)
    v_1_dropout = Dropout(0.2)(v_1)
    v_2 = Dense(5632//4, activation='relu')(v_1_dropout)
    v_3 = Dense(512, activation="relu")(v_2)
    v_3_1 = Dense(256, activation="relu")(v_3)
    v_3_2 = Dense(512, activation="relu")(v_3_1)
    v_4 = Dense(5632//4, activation='relu')(v_3_2)
    v_4_dropout = Dropout(0.2)(v_4)
    v_5 = Dense(5632//2, activation='relu')(v_4_dropout)
    v_6 = Dense(5632, activation='relu')(v_5)

    model = Model(inputs=v_input, outputs=v_6)
    return model

def v2v_128():
    v_input = Input(shape = (5632,), name='video_input')
    #v_input_noise = GaussianNoise(0.2)(v_input)
    v_1 = Dense(5632//2, activation='relu')(v_input)
    v_1_dropout = Dropout(0.2)(v_1)
    v_2 = Dense(5632//4, activation='relu')(v_1_dropout)
    v_3 = Dense(512, activation="relu")(v_2)
    v_3_1 = Dense(128, activation="relu")(v_3)
    v_3_2 = Dense(512, activation="relu")(v_3_1)
    v_4 = Dense(5632//4, activation='relu')(v_3_2)
    v_4_dropout = Dropout(0.2)(v_4)
    v_5 = Dense(5632//2, activation='relu')(v_4_dropout)
    v_6 = Dense(5632, activation='relu')(v_5)

    model = Model(inputs=v_input, outputs=v_6)
    return model

def v2v_64():
    v_input = Input(shape = (5632,), name='video_input')
    #v_input_noise = GaussianNoise(0.2)(v_input)
    v_1 = Dense(5632//2, activation='relu')(v_input)
    v_1_dropout = Dropout(0.2)(v_1)
    v_2 = Dense(5632//4, activation='relu')(v_1_dropout)
    v_3 = Dense(512, activation="relu")(v_2)
    v_3_1 = Dense(64, activation="relu")(v_3)
    v_3_2 = Dense(512, activation="relu")(v_3_1)
    v_4 = Dense(5632//4, activation='relu')(v_3_2)
    v_4_dropout = Dropout(0.2)(v_4)
    v_5 = Dense(5632//2, activation='relu')(v_4_dropout)
    v_6 = Dense(5632, activation='relu')(v_5)

    model = Model(inputs=v_input, outputs=v_6)
    return model

def a2a_512():
    a_input = Input(shape=(896,), name='audio_input')
    a_1 = Dense(896//2, activation='relu')(a_input)
    a_2 = Dense(512, activation='relu')(a_1)
    a_3 = Dense(896//2, activation="relu")(a_2)
    a_4 = Dense(896, activation='relu', name='audio_output')(a_3)
    model = Model(inputs=a_input, outputs=a_4)
    return model

def a2a_256():
    a_input = Input(shape=(896,), name='audio_input')
    a_1 = Dense(896//2, activation='relu')(a_input)
    a_2 = Dense(512, activation='relu')(a_1)
    a_2_1 = Dense(256, activation='relu')(a_2)
    a_2_2 = Dense(512, activation='relu')(a_2_1)
    a_3 = Dense(896//2, activation="relu")(a_2_2)
    a_4 = Dense(896, activation='relu', name='audio_output')(a_3)
    model = Model(inputs=a_input, outputs=a_4)
    return model

def a2a_128():
    a_input = Input(shape=(896,), name='audio_input')
    a_1 = Dense(896//2, activation='relu')(a_input)
    a_2 = Dense(512, activation='relu')(a_1)
    a_2_1 = Dense(128, activation='relu')(a_2)
    a_2_2 = Dense(512, activation='relu')(a_2_1)
    a_3 = Dense(896//2, activation="relu")(a_2_2)
    a_4 = Dense(896, activation='relu', name='audio_output')(a_3)
    model = Model(inputs=a_input, outputs=a_4)
    return model


def a2a_64():
    a_input = Input(shape=(896,), name='audio_input')
    a_1 = Dense(896//2, activation='relu')(a_input)
    a_2 = Dense(512, activation='relu')(a_1)
    a_2_1 = Dense(64, activation='relu')(a_2)
    a_2_2 = Dense(512, activation='relu')(a_2_1)
    a_3 = Dense(896//2, activation="relu")(a_2_2)
    a_4 = Dense(896, activation='relu', name='audio_output')(a_3)
    model = Model(inputs=a_input, outputs=a_4)
    return model

def joint_512():
    v_input = Input(shape=(5632,), name='video_input')
    a_input = Input(shape=(896,), name='audio_input') 

    # lets do a v-a decoding and a-v decoding and tie them up?
    # V to A
    v_1 = Dense(5632//2, activation='relu')(v_input)
    v_1d = Dropout(0.2)(v_1)
    v_2 = Dense(5632//4, activation='relu')(v_1d)
    v_2d = Dropout(0.2)(v_2)
    v_3 = Dense(5632//8, activation='relu')(v_2d)
    v_4 = Dense(512, activation="relu")(v_3)

    # A to V
    a_1 = Dense(5632//8, activation='relu')(a_input)
    a_2 = Dense(512, activation='relu')(a_1)

    joint_rep = concatenate([v_4, a_2])

    v_a_1 = Dense(5632//8, activation='relu')(joint_rep)
    a_output_from_v = Dense(7*128, activation='relu', name='decoded_audio')(v_a_1)

    a_v_3 = Dense(5632//8, activation='relu')(joint_rep)
    a_v_4 = Dense(5632//4, activation='relu')(a_v_3)
    a_v_4d = Dropout(0.2)(a_v_4)
    a_v_5 = Dense(5632//2, activation='relu')(a_v_4d)
    a_v_5d = Dropout(0.2)(a_v_5)
    v_output_from_a = Dense(11*512, activation='relu', name='decoded_video')(a_v_5d)

    model = Model(inputs=[v_input, a_input], outputs=[a_output_from_v, v_output_from_a])
    return model

def joint_256():
    v_input = Input(shape=(5632,), name='video_input')
    a_input = Input(shape=(896,), name='audio_input') 

    # lets do a v-a decoding and a-v decoding and tie them up?
    # V to A
    v_1 = Dense(5632//2, activation='relu')(v_input)
    v_1d = Dropout(0.2)(v_1)
    v_2 = Dense(5632//4, activation='relu')(v_1d)
    v_2d = Dropout(0.2)(v_2)
    v_3 = Dense(5632//8, activation='relu')(v_2d)
    v_4 = Dense(512, activation="relu")(v_3)
    v_5 = Dense(256, activation="relu")(v_4)

    # A to V
    a_1 = Dense(5632//8, activation='relu')(a_input)
    a_2 = Dense(512, activation='relu')(a_1)
    a_3 = Dense(256, activation='relu')(a_2)

    joint_rep = concatenate([v_5, a_3])

    v_a_1 = Dense(5632//8, activation='relu')(joint_rep)
    a_output_from_v = Dense(7*128, activation='relu', name='decoded_audio')(v_a_1)

    a_v_3 = Dense(5632//8, activation='relu')(joint_rep)
    a_v_4 = Dense(5632//4, activation='relu')(a_v_3)
    a_v_4d = Dropout(0.2)(a_v_4)
    a_v_5 = Dense(5632//2, activation='relu')(a_v_4d)
    a_v_5d = Dropout(0.2)(a_v_5)
    v_output_from_a = Dense(11*512, activation='relu', name='decoded_video')(a_v_5d)

    model = Model(inputs=[v_input, a_input], outputs=[a_output_from_v, v_output_from_a])
    return model

def joint_128():
    v_input = Input(shape=(5632,), name='video_input')
    a_input = Input(shape=(896,), name='audio_input') 

    # lets do a v-a decoding and a-v decoding and tie them up?
    # V to A
    v_1 = Dense(5632//2, activation='relu')(v_input)
    v_1d = Dropout(0.2)(v_1)
    v_2 = Dense(5632//4, activation='relu')(v_1d)
    v_2d = Dropout(0.2)(v_2)
    v_3 = Dense(5632//8, activation='relu')(v_2d)
    v_4 = Dense(512, activation="relu")(v_3)
    v_5 = Dense(128, activation="relu")(v_4)

    # A to V
    a_1 = Dense(5632//8, activation='relu')(a_input)
    a_2 = Dense(512, activation='relu')(a_1)
    a_3 = Dense(128, activation='relu')(a_2)

    joint_rep = concatenate([v_5, a_3])

    v_a_1 = Dense(5632//8, activation='relu')(joint_rep)
    a_output_from_v = Dense(7*128, activation='relu', name='decoded_audio')(v_a_1)

    a_v_3 = Dense(5632//8, activation='relu')(joint_rep)
    a_v_4 = Dense(5632//4, activation='relu')(a_v_3)
    a_v_4d = Dropout(0.2)(a_v_4)
    a_v_5 = Dense(5632//2, activation='relu')(a_v_4d)
    a_v_5d = Dropout(0.2)(a_v_5)
    v_output_from_a = Dense(11*512, activation='relu', name='decoded_video')(a_v_5d)

    model = Model(inputs=[v_input, a_input], outputs=[a_output_from_v, v_output_from_a])
    return model

def joint_64():
    v_input = Input(shape=(5632,), name='video_input')
    a_input = Input(shape=(896,), name='audio_input') 

    # lets do a v-a decoding and a-v decoding and tie them up?
    # V to A
    v_1 = Dense(5632//2, activation='relu')(v_input)
    v_1d = Dropout(0.2)(v_1)
    v_2 = Dense(5632//4, activation='relu')(v_1d)
    v_2d = Dropout(0.2)(v_2)
    v_3 = Dense(5632//8, activation='relu')(v_2d)
    v_4 = Dense(512, activation="relu")(v_3)
    v_5 = Dense(64, activation="relu")(v_4)

    # A to V
    a_1 = Dense(5632//8, activation='relu')(a_input)
    a_2 = Dense(512, activation='relu')(a_1)
    a_3 = Dense(64, activation='relu')(a_2)

    joint_rep = concatenate([v_5, a_3])

    v_a_1 = Dense(5632//8, activation='relu')(joint_rep)
    a_output_from_v = Dense(7*128, activation='relu', name='decoded_audio')(v_a_1)

    a_v_3 = Dense(5632//8, activation='relu')(joint_rep)
    a_v_4 = Dense(5632//4, activation='relu')(a_v_3)
    a_v_4d = Dropout(0.2)(a_v_4)
    a_v_5 = Dense(5632//2, activation='relu')(a_v_4d)
    a_v_5d = Dropout(0.2)(a_v_5)
    v_output_from_a = Dense(11*512, activation='relu', name='decoded_video')(a_v_5d)

    model = Model(inputs=[v_input, a_input], outputs=[a_output_from_v, v_output_from_a])
    return model

def joint_bidnn():
    v_input = Input(shape=(5632,), name='video_input')
    a_input = Input(shape=(896,), name='audio_input') 

    # lets do a v-a decoding and a-v decoding and tie them up?
    # V to A
    v_1 = Dense(5632//8, activation='relu')(v_input) 
    a_1 = Dense(5632//8, activation='relu')(a_input) 
    common_layer_1 = Dense(512, activation='relu')
    v_2 = common_layer_1(v_1)
    
    v_3 = Dense(512, activation="relu")(v_2)

    # A to V
    a_2 = common_layer_1(a_1)
    a_3 = Dense(512, activation='relu')(a_2)

    joint_rep = concatenate([v_3, a_3])

    common_layer_2 =  Dense(5632//8, activation='relu')

    v_a_1 = common_layer_2(joint_rep)
    a_output_from_v = Dense(7*128, activation='relu', name='decoded_audio')(v_a_1)

    a_v_3 = common_layer_2(joint_rep)
    v_output_from_a = Dense(11*512, activation='relu', name='decoded_video')(a_v_3)

    model = Model(inputs=[v_input, a_input], outputs=[a_output_from_v, v_output_from_a])
    return model

def joint_classical():
    v_input = Input(shape=(5632,), name='video_input')
    a_input = Input(shape=(896,), name='audio_input') 

    # lets do a v-a decoding and a-v decoding and tie them up?
    # V to A
    v_1 = Dense(5632//8, activation='relu')(v_input) 
    a_1 = Dense(5632//8, activation='relu')(a_input) 
    common_layer_1 = Dense(512, activation='relu')
    v_2 = common_layer_1(v_1)
    
    v_3 = Dense(5632//8, activation="relu")(v_2)

    # A to V
    a_2 = common_layer_1(a_1)
    a_3 = Dense(5632//4, activation='relu')(a_2)

    #joint_rep = concatenate([v_3, a_3])

    #common_layer_2 =  Dense(5632//8, activation='relu')

    #v_a_1 = common_layer_2(joint_rep)
    a_output_from_v = Dense(7*128, activation='relu', name='decoded_audio')(v_3)

    #a_v_3 = common_layer_2(joint_rep)
    v_output_from_a = Dense(11*512, activation='relu',
                            name='decoded_video')(a_3)

    model = Model(inputs=[v_input, a_input], outputs=[a_output_from_v, v_output_from_a])
    return model
