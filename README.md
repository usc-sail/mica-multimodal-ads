# Multimodal Representation of Advertisements Using Segment-level Autoencoders

This repository contains scripts to replicate the paper titled, "Multimodal Representation of Advertisements Using Segment-level Autoencoders" published at the 20th ACM International Conference on Multimodal Interaction, 2018. The paper can be found [here](https://drive.google.com/file/d/0B3ydbkt5jAUyUDRFLTN2Vi0wOUdoNEJ1ajE5Yl9hUk4xaVlr/view?usp=sharing)

The following python 2.7 packages need to be installed in order to run the scripts

```console
scipy==1.0.0
tqdm==4.11.2
numpy==1.11.0
pandas==0.22.0
six==1.10.0
tensorflow_gpu==1.4.0
joblib==0.11
Keras==2.1.5
scikit_learn==0.19.2
tensorflow==1.10.1
```

## Brief description of the scripts

1.   feature_extraction_scripts: C3D activity-net and vvgish features
2.   architectures.py: model configs for all experiments in the paper
3.   classification_experiments.py: classify using the embeddings
4.   create_baseline_experiments: replicate the experiments described [here](http://people.cs.pitt.edu/~kovashka/ads/)


In order to run these scripts you would need access to a few pickle files containining all the features. Email somandep@usc.edu to access these features.
