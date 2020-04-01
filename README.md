mnist: inference using saved pb file
====

Overview

1. Mnist deep learning using weight label like:

|Number| 0| 1| 2| 3| 4| 5| 6| 7| 8| 9|
|-|-|-|-|-|-|-|-|-|-|-|
|Label| 0.2| 0.8 | 0.2 |  0.2| 0.2 | 0.2| 0.2| 0.2| 0.2| 0.8|
Note: Answer is 1

2. Inference in Windows OS using pb file saved in Lenux 

This is the trial of:
- Train and save model in Lenux
- Convert the saved model into pb file
- Copy the pb file to Windows PC
- Inference in Windows OS

## Requirement

- python 3.5.2
- tensorflow 1.14.0

## Usage

1. save mnist raw data to your local like:  
./data/raw/t10k-images.idx3-ubyte  
./data/raw/t10k-labels.idx1-ubyte  
./data/raw/train-images.idx3-ubyte  
./data/raw/train-labels.idx1-ubyte  
1. start get_image.py
1. start configurate_data.py
1. start train.py
1. start freeze_graph.py
1. start apply_model_pb.py

Note: if you start apply_model.py without freeze_grapy, inferece is calculated from checkpoint.

## Licence

[MIT](https://github.com/tcnksm/tool/blob/master/LICENCE)
