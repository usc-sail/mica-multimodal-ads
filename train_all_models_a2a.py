import os, json, glob
import numpy as np
import keras
from keras.layers import *
from keras.models import Model
#from data_generator import *
from keras import metrics
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import pandas as pd
import tensorflow as tf
from keras import callbacks
from keras.regularizers import l2
from architectures import *

ads_fps = json.load( open('all_ads_fps.json', 'r') )

def _read_av_data_and_splice(ad_id, db_name_): #, vid_dir = './jwt_ads/c3d', aud_dir='./jwt_ads/vggish'):
    '''
    CHECK: 
        1. here fps changes for each ad_id, so I have a cvpr_ads_fps: global
           dict to look up ad_id for fps
        2. vid_dir, aud_dir are predefine: I find this easier to deal with

    input:
        ad_id: filename indicator: in the directory vid_dir, aud_dir
    
    output:
        vid_slice; shape(?,11,512): 5 features context both ways
        aud_slice: shape(?, 7, 128): 3 features context both ways
    '''
    f_id = ad_id.decode()
    db_name = db_name_.decode()
    #print(f_id)
    #f_loc = (np.argwhere(filenames==f_id)[0,0]).astype(np.int32)
    vid_file = os.path.join(db_name+'_ads', 'c3d', "%s.npz" % (f_id))
    aud_file = os.path.join(db_name+'_ads', 'vggish', "%s.npy" % (f_id))
    # squashing all into a GAP
    vid = np.mean(np.squeeze(np.load(vid_file)['conv']), (1,2,3))
    aud = np.load(aud_file)
    fps = ads_fps[f_id]['fps']
    # video duration per segment in frames
    vT = 16.0
    # audio duration per segment in secs
    aT = 0.96
    # window length and skip length [0...10], [3,...]
    vid_length=11
    vid_skip=3
    aud_length=7

    # get the indices for video frames with leaving out vid_skip windows
    # when I do [::vid_skip] i don't account for corner case at end so may lose
    # one sample window
    v_ix = [range(i,i+vid_length) for i in range(vid.shape[0]-vid_length+1)]
    v_ix = v_ix[::vid_skip]
    #print(v_ix)
    # work with the middle frame
    a_map = [int((i[5]*vT)/(fps*aT)) for i in v_ix]
    # since the aud length is 7 we need 3 fwd and bwd context
    # done manually maybe do this elegantly?
    a_ix = np.array([range(i-3, i+4) for i in a_map])
    
    # take care of corner cases while mapping usually off by one at edges
    a_ix[a_ix<0] = 0
    a_ix[a_ix>len(aud)-1] = len(aud)-1
    
    vid_slice = np.array([vid[i] for i in v_ix]).astype(np.float32)
    aud_slice = np.array([aud[i] for i in a_ix.tolist()]).astype(np.float32)
    # uncomment this to convince yourself that the suffle works as expected!
    #f_ = (np.zeros_like(aud_slice)+f_loc).astype(np.int32)
    return vid_slice, aud_slice#, f_ 

def _convert_splices_to_tensors(vid_slice_, aud_slice_):#, f_loc_):
    '''
    just convert the np arrays to tensors. important to explicitly mention
    dtype and shape. Need to merge this with the above function but not today
    '''

    vid_slice = tf.reshape(tf.convert_to_tensor(vid_slice_, tf.float32),
                           [-1,11*512])
    aud_slice = tf.reshape(tf.convert_to_tensor(aud_slice_, tf.float32),
                           [-1,7*128])
    #f_loc = tf.reshape(tf.convert_to_tensor(f_loc_, tf.int32), [-1,7,128])
    return vid_slice, aud_slice#, f_loc 

def ae_input_fn(db_name = 'jwt', batch_size=100, n_epochs=10, n_threads=16):
    if db_name == "jwt":
        df = pd.read_pickle('jwt_ads_data_N9744.pkl')
        filenames = np.array([str(i) for i in df['ad_id']])
    elif db_name == "cvpr":
        df = pd.read_pickle('cvpr_ads_data_labels_N2720.pkl')
        filenames = np.array([str(i) for i in df['files']]) 


    dataset = (tf.data.Dataset.from_tensor_slices(filenames) 
        .shuffle(buffer_size=10*len(filenames)) # shuffle filenames
        .repeat(-1) # the n_epochs makes sure you do sampling with rep
              )

    #format: https://www.tensorflow.org/api_docs/python/tf/py_func
    dataset = dataset.map(
        lambda ad_id: tuple(tf.py_func(_read_av_data_and_splice, [ad_id,db_name],
                                       [tf.float32, tf.float32] )),
                                        num_parallel_calls=n_threads)

    dataset = dataset.map(_convert_splices_to_tensors,
                          num_parallel_calls=20) # cpu-parallel

    # unbatch the slices produce - shuffle - batch - prefetch
    dataset = ( dataset.apply(tf.contrib.data.unbatch())
        .shuffle(buffer_size=100*batch_size) # make a big buffer to shuffle well
        .batch(batch_size)
        .prefetch(1)
              )

    # make a iter and generate!
    data_iter = dataset.make_one_shot_iterator()
    vid,aud = data_iter.get_next()
    #features = {'aud':aud, 'vid':vid}
    #target=[]
    return vid, aud #features, target

train_dataset = ae_input_fn(db_name="jwt")
test_dataset = ae_input_fn(db_name="cvpr")


model_256 = a2a_256()
model_128 = a2a_128()
model_64 = a2a_64()


model_256.compile( loss='mean_squared_error',optimizer='rmsprop',
              metrics=[metrics.mse, metrics.mae])

model_128.compile( loss='mean_squared_error',optimizer='rmsprop',
              metrics=[metrics.mse, metrics.mae])

model_64.compile( loss='mean_squared_error',optimizer='rmsprop',
              metrics=[metrics.mse, metrics.mae])

n_epochs = 30
batch_size = 100
n_train_steps = 495531//batch_size
n_test_steps = 668

sess = tf.InteractiveSession()

all_test_loss = []
for ep_ix in range(n_epochs):
    print(ep_ix, '-------------------------------------------')
    for b_ix in range(n_train_steps):
        _, aud = sess.run(train_dataset)
        if not b_ix%1000: verbosity = 2
        else: verbosity = 0
        model_256.fit(aud, aud, batch_size=100, epochs=1,
                         verbose=verbosity)
        
        model_128.fit(aud, aud, batch_size=100, epochs=1,
                         verbose=verbosity)
        model_64.fit(aud, aud, batch_size=100, epochs=1,
                         verbose=verbosity)
    # now test
    test_loss_list = []
    for b_test_ix in range(n_test_steps):
        _, aud_test = sess.run(test_dataset)
        test_loss = model_256.test_on_batch(aud_test, aud_test)
        test_loss_list.append(test_loss)
        test_loss = model_128.test_on_batch(aud_test, aud_test)
        test_loss_list.append(test_loss)
        test_loss = model_64.test_on_batch(aud_test, aud_test)
        test_loss_list.append(test_loss)
        #print(ep_ix, test_loss)
    all_test_loss.append(test_loss_list)
    print(ep_ix, ' mean loss: ',np.mean(test_loss_list), np.std(test_loss_list))
    if not ep_ix%10: model_256.save('a2a_256_ep_%d.h5' % ep_ix )
    if not ep_ix%10: model_128.save('a2a_128_ep_%d.h5' % ep_ix )
    if not ep_ix%10: model_64.save('a2a_64_ep_%d.h5' % ep_ix )
