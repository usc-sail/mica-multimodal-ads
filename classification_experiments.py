'''
@Krishna Somandepalli - 04/12/2018
1. preparing ads data with labels
2. Using saved fc6/fc7 features for replicating baselines for multiclass classification
'''
import os
import sys
import numpy as np
import json
import glob
from collections import Counter
import pandas as pd
from sklearn.model_selection import *
#cross_val_score, train_test_split, StratifiedKFold, GridSearchCV, KFold
from sklearn import datasets, svm, preprocessing
from collections import Counter
from sklearn.pipeline import make_pipeline, Pipeline
import tqdm
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
import warnings
from scipy.io import loadmat, savemat
warnings.filterwarnings("ignore") # not condoned - just for pretty sake

# listt of ads - a text file with the fullpaths of the ads considered
ads_images_file = "./ads_images_list.txt"
# where be all the labels
label_file_dir = "../data/annotations/video/cleaned_result/"
do_PCA = False


def run_svc_pipeline_doubleCV(X, y, dev_split=5, C_=0.015, n_splits_=10, param_search=True, n_jobs_=18): 
    # use different splits with different random states for CV-param search
    svc = svm.SVC(kernel='linear', C = C_) 
    #svc = svm.LinearSVC(C = C_) 

    pipeline_estimators = [('scale', preprocessing.MinMaxScaler()), 
                           ('svm', svc) ]
    #pipeline_estimators = [('svm', svc)]
    svc_pipeline = Pipeline(pipeline_estimators)

    if param_search:
        C_search = sorted( list(np.logspace(-5,0,10)) + [0.1,5,10,20,50,100] )
        param_grid = dict( scale=[None], svm__C=C_search )
        #param_grid = dict( svm__C=C_search )

        sk_folds = StratifiedKFold(n_splits=dev_split, shuffle=False,
                                   random_state=1964)
        grid_search = GridSearchCV(svc_pipeline, param_grid=param_grid,
                                   n_jobs=n_jobs_, cv=sk_folds.split(X,y),
                                   verbose=False)
        grid_search.fit(X, y)
        # find the best C value
        which_C = np.argmax(grid_search.cv_results_['mean_test_score'])
        best_C = C_search[which_C]
    else:
        best_C = C_

    svc_pipeline.named_steps['svm'].C = best_C
    #print('estimated the best C for svm to be', best_C)
    sk_folds = StratifiedKFold(n_splits=n_splits_, shuffle=False, random_state=320)
    all_scores = []
    all_y_test = []
    all_pred = []
    for train_index, test_index in sk_folds.split(X, y):
#        print 'run -',
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        svc_pipeline.fit(X_train, y_train)
        y_pred = svc_pipeline.predict(X_test)
        score = svc_pipeline.score(X_test, y_test)
 #       print score	
        all_y_test.append(y_test)
        all_pred.append(y_pred)
        all_scores.append(score)
    return all_y_test, all_pred, all_scores

def classify_prominent_groups_only(X, y, n_prominent = 10):
    # n_prominent: min. number of classes in a group for things to be
    # classified
    y_count = Counter(y)
    label_gt10 = [ i for i,j in y_count.items() if j>n_prominent ]
    label_mask = [True if i in label_gt10 else False for i in y]
    y_gt10 = y[label_mask]
    X_gt10 = X[label_mask]
    print('data shape:', X_gt10.shape,'label_shape:', y_gt10.shape)
    y_test_gt10, y_pred_gt10, y_scores_gt10 = run_svc_pipeline(X_gt10, y_gt10)
    print( 'acc/score: ', np.mean(y_scores_gt10) )

    return y_test_gt10, y_pred_gt10, y_scores_gt10

# LOAD DATA
ads_data_label_file = "cvpr_ads_data_labels_N2720.pkl"
if not os.path.isfile(ads_data_label_file): 
    df = prepare_data(ads_images_file, label_file_dir, \
                      SAVE_pickle = True, pkl_name=ads_data_label_file)
else: df = pd.read_pickle(ads_data_label_file)
#df_dict=df.to_dict()

filenames = np.array(df.files)

if not os.path.isfile('cvpr_mean_av_embeddings.npz'):
    joint_features = {}
    for j_ in ['joint_v2a', 'a2a', 'joint_a2v', 'v2v']:
        print(j_)
        df1 = pd.read_pickle('cvpr_%s_embeddings.pkl' % j_)
        joint_features[j_] = np.array(
            [np.mean(df1[i_][0],0) for ix_,i_ in enumerate(filenames)
            if cca_label_mask[ix_]])
    np.savez('cvpr_max_av_embeddings', joint_v2a=joint_features['joint_v2a'],
             joint_a2v=joint_features['joint_a2v'],
             a2a=joint_features['a2a'], v2v=joint_features['v2v'])

print('loading joint files')
joint_features = np.load('cvpr_mean_av_embeddings.npz')
X_joint = np.hstack([joint_features[i] for i in ['joint_v2a', 'a2a',
                                                 'joint_a2v', 'v2v']])

COLLAPSE = False 
if COLLAPSE:
    X_cca = np.sum([X_cca[:,512*i:512*(i+1)] for i in range(4)], axis=0)
    X_joint = np.sum([X_joint[:,512*i:512*(i+1)] for i in range(4)], axis=0)

X_text_ = np.load('cvpr_word_embeddings.npy')
X_text = X_text_[cca_label_mask]
X_cca_text = np.hstack([X_cca, X_text])
X_joint_text = np.hstack([X_joint, X_text])



if do_PCA:
    print('PCA---')
    P = PCA(whiten=True)
    X_joint = P.fit_transform(X_joint)[:,:16]
    X_cc = P.fit_transform(X_cca)[:,:16]

# manipulate Funny labels as per the CVPR paper - text 
if True:
    attr_names = ['Exciting', 'Funny']
    for mode, X, mask_ in [('cca', X_cca, cca_label_mask), 
                           ('text_cca', X_cca_text, cca_label_mask),  
                           ('joint_text', X_joint_text, cca_label_mask),
                          ('joint', X_joint, cca_label_mask)]:
        for attr_name in attr_names:
            y_attr = df[attr_name][mask_]
            lup_ix = list((y_attr<=0.3) + (y_attr>=0.7))
            y_mapped = np.array(y_attr[lup_ix])
            y_mapped[y_mapped<=0.3] = 0
            y_mapped[y_mapped>=0.7] = 1
            y_mapped = y_mapped.astype('int')

            #[('aud', X_audio[lup_ix]), ('vid', X_video[lup_ix])]:
            print(mode, '--------------------------------------------------')
            print(attr_name,': data shape:', X[lup_ix].shape,'label_shape:', y_mapped.shape)
            y_test, y_pred, y_scores = run_svc_pipeline_doubleCV(X[lup_ix], y_mapped)
            print( attr_name, ': accuracy/score: ',
                  np.mean(y_scores),np.std(y_scores) )

# BAselines for 3 problems
if True:
    attr_names = ['Sentiments', 'Topics', 'Effective'][::-1]
    ## do some CV testing
    for attr_name in attr_names:
        y = np.array(df[attr_name]) 
        for mode, X, y in [('cca', X_cca, cca_label_mask), 
                               ('text_cca', X_cca_text, cca_label_mask),  
                               ('joint_text', X_joint_text, cca_label_mask),
                              ('joint', X_joint, cca_label_mask)]:
            print(mode, '--------------------------------------------------')
            print(attr_name,': data shape:', X.shape,'label_shape:',
                  y[cca_label_mask].shape)
            y_test, y_pred, y_scores = run_svc_pipeline_doubleCV(
                X, y[cca_label_mask])#, n_splits_=5, n_jobs_=10)
            print( attr_name, ': accuracy/score: ',
                  np.mean(y_scores),np.std(y_scores) )
            


