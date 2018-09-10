#!/bin/bash

# first download https://github.com/usc-sail/mica-gender-from-audio.git to a dir say /data/

## Usage:
# ./get_audio_set_features.sh <wav-file-list> <output-dir> <num-cores>
# bash get_audioset_features.sh list_of_wav_files.txt /data/features 4

proj_dir=/data/mica-gender-from-audio
py_scripts_dir=/data/mica-gender-from-audio/python_scripts
wav_list=${1} # full paths
feats_dir=${2}
nj=${3}
mkdir -p ${feats_dir}
movie_count=1
for wav_file in `cat ${wav_list}`
do
    python $py_scripts_dir/compute_and_write_vggish_feats.py $proj_dir $wav_file $feats_dir/vggish &
    if [ $(($movie_count % $nj)) -eq 0 ]; then
        wait
    fi
    movie_count=`expr $movie_count + 1`
done
wait

