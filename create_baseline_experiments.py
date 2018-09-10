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
warnings.filterwarnings("ignore") # not condoned - just for pretty sake



# listt of ads - a text file with the fullpaths of the ads considered
ads_images_file = "./ads_images_list.txt"
# where be all the labels
label_file_dir = "../data/annotations/video/cleaned_result/"

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

# do
def run_svc_pipeline(X, y, C_=0.1, n_splits_=5, param_search=False, n_jobs_=18): 
    svc = svm.SVC(kernel='linear', C = C_) 
    #svc = svm.LinearSVC(C = C_) 

    pipeline_estimators = [('scale', preprocessing.StandardScaler()), \
                           ('svm', svc)]
    svc_pipeline = Pipeline(pipeline_estimators)

    if param_search:
        param_grid = dict( scale=[None], 
                          svm__C=sorted(list(np.logspace(-5,0,10))+[0.1,5,10,20,50,100]) )

        grid_search = GridSearchCV(svc_pipeline, param_grid=param_grid, \
                                   n_jobs=n_jobs_, cv=n_splits_, verbose=True)
        grid_search.fit(X, y)
        
        return grid_search
    else:
        sk_folds = StratifiedKFold(n_splits=n_splits_ , shuffle=True, random_state=320)
        all_scores = []
        all_y_test = []
        all_pred = []
        for train_index, test_index in sk_folds.split(X, y):
            print 'run -',
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            svc_pipeline.fit(X_train, y_train)
            y_pred = svc_pipeline.predict(X_test)
            score = svc_pipeline.score(X_test, y_test)
            print score	
            all_y_test.append(y_test)
            all_pred.append(y_pred)
            all_scores.append(score) 
        return all_y_test, all_pred, all_scores

def run_svc_pipeline_dev(X, y, dev_split=5, C_=0.015, n_splits_=10, param_search=True, n_jobs_=18): 
    # dev on a held out partition learn the params and then test on the rest
    # the partitions
    svc = svm.SVC(kernel='linear', C = C_) 
    #svc = svm.LinearSVC(C = C_) 

    pipeline_estimators = [('scale', preprocessing.StandardScaler()), ('svm',
                                                                       svc) ]
    svc_pipeline = Pipeline(pipeline_estimators)

    # make 3 splits use 1 for dev and the rest to train/test using CV
    kf = KFold(n_splits=dev_split, random_state=320)
    k1 = kf.split(X,y)
    train_ix, dev_ix = k1.next()
    # third split becomes dev
    X_, X_dev = X[train_ix], X[dev_ix]
    y_, y_dev = y[train_ix], y[dev_ix]
    
    print('dev\'ing on data: ', y_dev.shape)
    print('CV\'ing on data: ', y_.shape)
    
    if param_search:
        C_search = sorted( list(np.logspace(-5,0,10)) + [0.1,5,10,20,50,100] )
        param_grid = dict( scale=[None], svm__C=C_search )

        grid_search = GridSearchCV(svc_pipeline, param_grid=param_grid, \
                                   n_jobs=n_jobs_, cv=1, verbose=True)
        grid_search.fit(X_dev, y_dev)
        # find the best C value
        which_C = np.argmax(grid_search.cv_results_['mean_test_score'])
        best_C = C_search[which_C]
    else:
        X_, y_ = X,y
        best_C = C_

    svc_pipeline.named_steps['svm'].C = best_C
    print('estimated the best C for svm to be', best_C)
    sk_folds = StratifiedKFold(n_splits=n_splits_, shuffle=True, random_state=320)
    all_scores = []
    all_y_test = []
    all_pred = []
    for train_index, test_index in sk_folds.split(X_, y_):
        print 'run -',
        X_train, X_test = X_[train_index], X_[test_index]
        y_train, y_test = y_[train_index], y_[test_index]

        svc_pipeline.fit(X_train, y_train)
        y_pred = svc_pipeline.predict(X_test)
        score = svc_pipeline.score(X_test, y_test)
        print score	
        all_y_test.append(y_test)
        all_pred.append(y_pred)
        all_scores.append(score)
    return all_y_test, all_pred, all_scores

def run_svc_pipeline_doubleCV(X, y, dev_split=5, C_=0.015, n_splits_=10, param_search=True, n_jobs_=18): 
    # use different splits with different random states for CV-param search
    svc = svm.SVC(kernel='linear', C = C_) 
    #svc = svm.LinearSVC(C = C_) 

    pipeline_estimators = [('scale', preprocessing.StandardScaler()), ('svm',
                                                                       svc) ]
    svc_pipeline = Pipeline(pipeline_estimators)

    if param_search:
        C_search = sorted( list(np.logspace(-5,0,10)) + [0.1,5,10,20,50,100] )
        param_grid = dict( scale=[None], svm__C=C_search )

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

def generate_avg_c3d_feats(out_file_name = 'mean_c3d_features.npz'):
    vid_feat_dir = './c3d_features/'
    all_fc1 = []
    all_fc2 = []
    for file_i in tqdm.tqdm(df.files):
        f_path = os.path.join(vid_feat_dir, file_i+'.npz')
        f_obj = np.load(f_path)
        all_fc1.append( np.mean( np.squeeze(f_obj['fc1']) , 0) )
        all_fc2.append( np.mean( np.squeeze(f_obj['fc2']) , 0) )

    np.savez(out_file_name, files=df.files, fc1=all_fc1, fc2=all_fc2)
    return 0

def generate_avg_vggish_feats(out_file_name = 'mean_vggish_features.npz'):
    vid_feat_dir = './vggish/'
    all_fc = []
    for file_i in tqdm.tqdm(df.files):
        f_path = os.path.join(vid_feat_dir, file_i+'.npy')
        f_obj = np.load(f_path)
        all_fc.append( np.mean(f_obj, 0) )
    
    np.savez(out_file_name, files=df.files, fc=all_fc)

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
df_dict=  df.to_dict()


### - create video features
# load the mean c3d (video feats) file
mean_c3d_feats_file = 'cvpr_mean_c3d_features.npz'
if not os.path.isfile(mean_c3d_feats_file): 
    generate_avg_c3d_feats(out_file_name = mean_c3d_feats_file)

mean_c3d_feats = np.load(mean_c3d_feats_file) 
#X_video = mean_c3d_feats['fc2']

### - create video features
# load the mean c3d (video feats) file
gap_c3d_feats_file = 'cvpr_GAP_512dim_c3d_features.npz'
if not os.path.isfile(gap_c3d_feats_file): 
    generate_GAP_c3d_feats(out_file_name = gap_c3d_feats_file)

gap_c3d_feats = np.load(gap_c3d_feats_file)
X_video_ = gap_c3d_feats['gap']
X_video = []
for i in X_video_:
    X_video.append( np.mean(i,0) )
X_video = np.array(X_video)

PCA_ = False
if PCA_:
    print('doing PCA!')
    vid_pca = PCA(n_components=128)
    X_video_pca = vid_pca.fit_transform(X_video)
# SVC: - all labels, using 10 folds, but have some classes with less than 10 

### - create audio features
# load the mean vggish (audio feats) file
mean_vggish_feats_file = 'cvpr_mean_vggish_features.npz'
if not os.path.isfile(mean_vggish_feats_file): 
    generate_avg_vggish_feats(out_file_name = mean_vggish_feats_file)

mean_vggish_feats = np.load(mean_vggish_feats_file) 
X_audio = mean_vggish_feats['fc']


### Text baselines:
# need atleast one of title or cc to predict
# text=titles; cc=subtitles; text=mean(cc, titles)
if True:
    text_df = pd.read_pickle('cvpr_ads_text_embeddings.pkl')
    title_embeddings = text_df['embedding_titles_mean_norm']
    cc_embeddings = text_df['embedding_cc_mean_norm']
    

    title_label_mask = [ True if len(i)>0 else False for i in title_embeddings ]
    print('No. available ads with usable title info: ', sum(title_label_mask))
    X_titles = np.array([ i for i in title_embeddings if len(i)>0 ] )

    cc_label_mask = [ not(np.isnan(i).any()) for i in cc_embeddings ]
    print('No. available ads with NOISY cc info: ', sum(cc_label_mask))
    X_cc = np.array([i for i in cc_embeddings if not np.isnan(i).any()])

    text_label_mask = (np.array(title_label_mask) &
                       np.array(cc_label_mask)).tolist()

    print('No. available ads with both cc and titles: ', sum(text_label_mask))
    X_text = np.array([np.hstack([i,j]) for i,j in zip(title_embeddings, cc_embeddings) if
                       (len(i)>0 and not np.isnan(j).any()) ])
 

# BAselines for 3 problems
if True:
    attr_names = ['Sentiments', 'Topics', 'Effective']
    ## do some CV testing
    for attr_name in attr_names:
        y = np.array(df[attr_name]) 
        for mode, X, y in [ ('aud', X_audio, y), ('vid', X_video,y),
                           ('title', X_titles, y[title_label_mask]), 
                            ('cc', X_cc, y[cc_label_mask]),
                            ('title_cc', X_text, y[text_label_mask]) ]:
            print(mode, '--------------------------------------------------')
            print(attr_name,': data shape:', X.shape,'label_shape:', y.shape)
            y_test, y_pred, y_scores = run_svc_pipeline_doubleCV(X, y)#, n_splits_=5, n_jobs_=10)
            print( attr_name, ': accuracy/score: ', np.mean(y_scores) )
            


# transform Funny labels as per the CVPR paper -A/V 
if False:
    attr_names = ['Exciting', 'Funny']

    for attr_name in attr_names:
        y_attr = df[attr_name]
        lup_ix = list((y_attr<=0.3) + (y_attr>=0.7))
        y_mapped = np.array(y_attr[lup_ix])
        y_mapped[y_mapped<=0.3] = 0
        y_mapped[y_mapped>=0.7] = 1
        y_mapped = y_mapped.astype('int')


        for mode, X, y_mapped  in [('text', X_titles[lup_ix], y_mapped)]:
            #[('aud', X_audio[lup_ix]), ('vid', X_video[lup_ix])]:
            print(mode, '--------------------------------------------------')
            print(attr_name,': data shape:', X.shape,'label_shape:', y_mapped.shape)
            y_test, y_pred, y_scores = run_svc_pipeline_doubleCV(X, y_mapped)#, n_splits_=5, n_jobs_=10)
            print( attr_name, ': accuracy/score: ', np.mean(y_scores) )

# transform Funny labels as per the CVPR paper - text 
if True:
    attr_names = ['Exciting', 'Funny']
    all_mask = [True for i in range(len(X_audio))]
    for mode, X, mask_ in [ ('titles', X_titles, title_label_mask), 
                        ('cc', X_cc, cc_label_mask),
                        ('title_cc', X_text, text_label_mask),
                        ('aud', X_audio, all_mask), 
                        ('vid', X_video, all_mask) ]:
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
            print( attr_name, ': accuracy/score: ', np.mean(y_scores) )

