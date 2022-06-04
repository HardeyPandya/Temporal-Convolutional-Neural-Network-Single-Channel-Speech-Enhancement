# Temporal-Convolutional-Neural-Network-Single-Channel-Speech-Enhancement
Code Segments of overlapping frames and Overlap and Add is taken from 
https://github.com/ashutosh620/DDAEC


TCNN model architecture taken from https://github.com/LXP-Never/TCNN

Baseline Training and inference scripts are taken from with necessary changes(converting code from STFT domain to overlapping frames)
https://github.com/haoxiangsnr/IRM-based-Speech-Enhancement-using-LSTM/blob/master/train.py

Thanks to all those who are making their codes open source - promoting education, learning and research.

**Guide to run these files on Google Colab**
From train.py file, model training and validation scripts are called to run.
Model architecture is there in model/tcnn.py file

Dataset file paths are stored in audioNOISE.txt and audioTRAIN.txt(path for clean speech files) files. You need to prepare your own file paths to train the model on your preferred dataset. 

You need clean speech and noise datasets to train the model. Model learns to denoise the given input audio file. So, noisy file is given as input to the model(feature) and clean speech file is corresponding output(also called label). 

For preparing noisy speech, we mix the clean speech file with noise to generate noisy speech file. Network is trained by calculating MSE loss between this noisy speech and clean speech. 
