import os, sys, glob, cv2
import numpy as np
import json
from PIL import Image
# where are the DNN models, etc?
import sys
dnn_dir = "/proj/krishna/ads/DNNs/C3D-tensorflow/"
sys.path.insert(0, dnn_dir)
import os
import os.path
import tensorflow as tf
import c3d_model
import pickle
import tqdm
from collections import Counter, deque
from itertools import islice
#os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def read_and_decode(example):
    # decoding tfr files from vvgish - audioset
    context_features = {'movie_id': tf.FixedLenFeature([], tf.string)}
    sequence_features = {'audio_embedding': tf.FixedLenSequenceFeature([], tf.string)}

    context_parsed, sequence_parsed = tf.parse_single_sequence_example(example, 
        context_features = context_features, sequence_features = sequence_features)

    normalized_feature = tf.divide(
                tf.decode_raw(sequence_parsed['audio_embedding'], tf.uint8),
                tf.constant(255, tf.uint8))
    shaped_feature = tf.reshape(tf.cast(normalized_feature, tf.float32),
                                    [-1, 128])
    
    return context_parsed['movie_id'], shaped_feature

def convert_tfr_to_np(tfr_dir, out_dir, filename=None):
    if filename is None: tfr_list = glob.glob(tfr_dir+'/*record')
    else:
        tfr_list = [os.path.join(tfr_dir, i.strip()+'.tfrecord') for i in
                    open(filename, 'r').readlines()]

	sess = tf.InteractiveSession()
	#record_iter = tf.python_io.tf_record_iterator(path = tfr_list)
	for tfr_i in tqdm.tqdm(tfr_list):
	    #print(tfr_i)
	    tfr_iter = tf.python_io.tf_record_iterator(path = tfr_i)
	    id_, x = read_and_decode( tfr_iter.next() )
	    id_name = id_.eval()
	    x_arr = x.eval()
	    out_name = os.path.join(out_dir, id_name+'.npy')
	    np.save(out_name, x_arr)    

def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the input tensors.
  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.
  Args:
    batch_size: The batch size will be baked into both placeholders.
  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         c3d_model.NUM_FRAMES_PER_CLIP,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CHANNELS))
  labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size))
  return images_placeholder, labels_placeholder


# pretrained mean file:
np_mean = np.load("/proj/krishna/ads/DNNs/C3D-tensorflow/crop_mean.npy")
#.reshape([num_frames_per_clip, crop_size, crop_size, 3])

def preprocess_image(frame_list, mean_file = np_mean, crop_size = 112):
	im_list = []

	for im in frame_list:
		img = Image.fromarray( cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.uint8) )
		if(img.width>img.height):
			scale = float(crop_size)/float(img.height)
			img = np.array( cv2.resize(np.array(img), \
				(int(img.width * scale + 1), crop_size) ) ).astype(np.float32)
		else:
			scale = float(crop_size)/float(img.width)
			img = np.array(cv2.resize(np.array(img),(crop_size, \
					int(img.height * scale + 1)))).astype(np.float32)
		crop_x = int((img.shape[0] - crop_size)/2)
		crop_y = int((img.shape[1] - crop_size)/2)
		img = img[crop_x:crop_x+crop_size, crop_y:crop_y+crop_size,:]
		im_list.append(img)

	img_arr = (np.array(im_list) - np_mean).astype(np.float32)

	return img_arr


def prepare_data(ads_images_file, label_file_dir, SAVE_pickle = True, pkl_name = '/tmp/tmp.pkl'):
    '''
    inputs:
        ads_images_file: a text file with each line for a unique identifier of the ad
        label_file_dir: directory with label json files "label_Effective_*.json"

    outputs:
        a dataframe saved as a pickle
        returns a dataframe (df)
        '''

    ad_list = [i.strip() for i in open(ads_images_file, 'r').readlines()]
    ad_names = [os.path.basename(i) for i in ad_list]

    label_files = glob.glob(label_file_dir + "/*.json")
    label_dict = {}

    # get all the labels in a giant dict
    for l in label_files:
        label_name = os.path.basename(l).split('.')[0].split('_')[1]
        label_dict[label_name] = json.load(open(l, 'r'))
        label_names = label_dict.keys()

    data_labels_dict = {}
    data_labels_dict['files'] = ad_names
    for l in label_names:
        l_keys = label_dict[l].keys()
        data_labels_dict[l] = [label_dict[l][i] if i in l_keys else '' \
                               for i in ad_names]

    df = pd.DataFrame(data = data_labels_dict)
    if SAVE_pickle: 
        df.to_picle(ads_data_label_file)
        #df.to_pickle("ads_data_labels_N%d.pkl" % (len(ad_names)))
        return df

def windowed(iterable, n=2):
    # from stackoverflow answers
    it = iter(iterable)
    win = deque(islice(it, n), n)
    if len(win) < n:
        return
    append = win.append
    yield list(win)
    for e in it:
        append(e)
        yield list(win)

cvpr_ads_fps = json.load( open('cvpr_ads_fps.json', 'r') )
def generate_frame_level_splices(ad_id, vid_dir = './cvpr_ads/c3d',
                                 aud_dir='./cvpr_ads/vggish', fps=23.98,
                                 vid_length=11, vid_skip=3, aud_length=7 ):
    '''
    vid_length: splice duration for video features
    vid_skip: how many to skip for the next sample

    determine aud duration from fps and pick out the audio middle frame with 3
    samples on either side for context accordingly
    '''
    # video duration per segment in frames
    vT = 16.0
    # audio duration per segment in secs
    aT = 0.96
    fps = cvpr_ads_fps[ad_id]
    # squashing all into a GAP
    vid_file = os.path.join(vid_dir,'%s.npz' % (ad_id))
    aud_file = os.path.join(aud_dir, '%s.npy' % (ad_id))
    vid = np.mean(np.squeeze(np.load(vid_file)['conv']), (1,2,3))
    aud = np.load(aud_file)

    # get the indices for video frames with leaving out vid_skip windows
    # when I do [::vid_skip] i don't account for corner case at end so may lose
    # one sample window
    v_ix = [i for i in windowed(range(vid.shape[0]),vid_length)][::vid_skip]
    # work with the middle frame
    a_map = [int((i[5]*vT)/(fps*aT)) for i in v_ix]
    # since the aud length is 7 we need 3 fwd and bwd context
    a_ix = np.array([range(i-3, i+4) for i in a_map])
    
    # take care of corner cases while mapping usually off by one at start or
    # end
    a_ix[a_ix<0] = 0
    a_ix[a_ix>len(aud)-1] = len(aud)-1

    vid_slice = np.array([vid[i] for i in v_ix])
    aud_slice = np.array([aud[i] for i in a_ix.tolist()])

    return vid_slice, aud_slice



