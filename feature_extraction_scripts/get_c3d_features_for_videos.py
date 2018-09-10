# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#---
# Edited by KS 04/18 to read a giant list of video files and write out features for every 16 frames
# TO FIX : it throws away the last accumulator if #frames <16, fix this by some random sampling
#----

'''Usage
python get_c3d_features_for_videos list_of_video_filepaths.txt

this take a list of video files and outputs features every 16 frames : a single
npz file for every video

Download the pretrained models from https://github.com/hx173149/C3D-tensorflow.git
and set this param:
dnn_dir = /path/to/the/c3d-dir
'''


# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# where are the DNN models, etc?
import sys
dnn_dir = "/data/ads/DNNs/C3D-tensorflow/"
sys.path.insert(0, dnn_dir)

import os
import os.path
import tensorflow as tf
import c3d_model
import numpy as np
import pickle
from utils import * 

# get a list of all video files and do some sorting?
all_videos = [i.strip() for i in open(sys.argv[1], 'r').readlines()] #glob.glob('../data/jwt_ads/videos/*.mp4')
# out_dir: where be the output files?
out_dir = "./jwt_ads/c3d_features"
if not os.path.isdir(out_dir): os.makedirs(out_dir)


# Basic model parameters as external flags.
flags = tf.app.flags
#gpu_num = 1
gpu_id = str(sys.argv[2])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
flags.DEFINE_integer('batch_size', 1 , 'Batch size.')
FLAGS = flags.FLAGS

# ---------------------------------------------------------------------------------------
def _variable_on_cpu(name, shape, initializer):
  #with tf.device('/cpu:%d' % cpu_id):
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
  if wd is not None:
    weight_decay = tf.nn.l2_loss(var) * wd
    tf.add_to_collection('losses', weight_decay)
  return var


def run_test():
  model_name = os.path.join(dnn_dir, "models", "sports1m_finetuning_ucf101.model")

  # Get the sets of images and labels for training, validation, and
  images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)
  with tf.variable_scope('var_name') as var_scope:
    weights = {
            'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.04, 0.00),
            'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.04, 0.00),
            'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.04, 0.00),
            'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.04, 0.00),
            'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.04, 0.00),
            'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wd1': _variable_with_weight_decay('wd1', [8192, 4096], 0.04, 0.001),
            'wd2': _variable_with_weight_decay('wd2', [4096, 4096], 0.04, 0.002),
            'out': _variable_with_weight_decay('wout', [4096, c3d_model.NUM_CLASSES], 0.04, 0.005)
            }
    biases = {
            'bc1': _variable_with_weight_decay('bc1', [64], 0.04, 0.0),
            'bc2': _variable_with_weight_decay('bc2', [128], 0.04, 0.0),
            'bc3a': _variable_with_weight_decay('bc3a', [256], 0.04, 0.0),
            'bc3b': _variable_with_weight_decay('bc3b', [256], 0.04, 0.0),
            'bc4a': _variable_with_weight_decay('bc4a', [512], 0.04, 0.0),
            'bc4b': _variable_with_weight_decay('bc4b', [512], 0.04, 0.0),
            'bc5a': _variable_with_weight_decay('bc5a', [512], 0.04, 0.0),
            'bc5b': _variable_with_weight_decay('bc5b', [512], 0.04, 0.0),
            'bd1': _variable_with_weight_decay('bd1', [4096], 0.04, 0.0),
            'bd2': _variable_with_weight_decay('bd2', [4096], 0.04, 0.0),
            'out': _variable_with_weight_decay('bout', [c3d_model.NUM_CLASSES], 0.04, 0.0),
            }
  ## append logits/classification across gpus
  #for gpu_index in range(0, gpu_num):
  with tf.device('/gpu:%s' % gpu_id):
      logits = c3d_model.inference_c3d(images_placeholder, 0.6, FLAGS.batch_size, weights, biases)
  #logits = tf.concat(logits,0)
  #norm_score = tf.nn.softmax(logits)

# saved version of the model
  saver = tf.train.Saver()
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  # init vars
  init = tf.global_variables_initializer()
  sess.run(init)
  # Create a saver for writing training checkpoints.
  saver.restore(sess, model_name)

  # obtain the last conv layer before pooling - this allows to do aptial/spatio-temporal pooling in future
  embeddings = sess.graph.get_tensor_by_name("relu5b:0")
  # get fc1
  fc_1 = sess.graph.get_tensor_by_name("fc1:0") 
  fc_2 = sess.graph.get_tensor_by_name("fc2:0") 


  for vid_file in all_videos:

    
    vid_name = os.path.basename(vid_file).split('.')[0]
    
    all_fc1 = []
    all_fc2 = []
    all_conv = []
    frame_list = []
    
    vid = cv2.VideoCapture(vid_file)
                
    frame_c = 0
    seg_count = 0
    print(vid_name, '------')
    # start reading vid file
    while vid.isOpened():
      ret, frame = vid.read()
      if frame is not None:
        if not frame_c%16: frame_list.append(frame)
        else:
          frame_list.append(frame)
          # when 16 frames are accumulated - get features save them to a list - and reset frame_list
          if len(frame_list) == 16:
            seg_count += 1
            if not seg_count%10: print(seg_count)
            test_images = preprocess_image(frame_list)[np.newaxis, ...]
            
            # extract all features
            conv_output = embeddings.eval(session = sess, 
                                        feed_dict = {images_placeholder: test_images})
    
            fc1_output = fc_1.eval(session = sess, 
                                        feed_dict = {images_placeholder: test_images})
            
            fc2_output = fc_2.eval(session = sess, 
                                        feed_dict = {images_placeholder: test_images})


            all_conv.append(conv_output)
            all_fc1.append(fc1_output)
            all_fc2.append(fc2_output)
            
            # done extracting - reset the frame "accumulator"
            frame_list = []

        frame_c += 1
     
      else:
        # release video data gen.
        vid.release()
        # save the file
        np.savez(os.path.join(out_dir, vid_name), conv = all_conv, fc1 = all_fc1, fc2 = all_fc2)
  print("done")

def main(_):
  run_test()

if __name__ == '__main__':
  tf.app.run()
