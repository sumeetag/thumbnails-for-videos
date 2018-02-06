# Auto selection of thumbnails for videos.

**A thumbnail worth a thousand frames**

Deep Neural Network - Automatic selection of the most optimal and relevant Thumbnails for Videos.

This repo provides the package that can select top 3 most **optimal and most relevant thumbnails** for any given video. Here are a few example outputs:

<img src='imgs/result1.png'>

This package provides - 
- A pretrained model
- Code to run the model on new videos, on either CPU or GPU
- Instructions for training the model

## Luarocks Installation

luarocks install torch
luarocks install nn
luarocks install image
luarocks install lua-cjson
luarocks install https://raw.githubusercontent.com/qassemoquab/stnbhwd/master/stnbhwd-scm-1.rockspec
luarocks install https://raw.githubusercontent.com/jcjohnson/torch-rnn/master/torch-rnn-scm-1.rockspec

luarocks install cutorch
luarocks install cunn
luarocks install cudnn

## PIP Installation

cv2 - pip install opencv-python
matplotlib.pyplot - pip install matplotlib
PIL - pip install pillow
spacy - pip install -U spacy
spacy - python -m spacy download en


## Pretrained model

You can download a pretrained CNN Binary Classifier model by running the following link:

[https://drive.google.com/drive/u/1/folders/1vtrfy1sGQM5SUqdN85LRhr4qXD8Tiu3L]

This will download a zipped version of the CNN model (about 1.5 GB)

You can download a pretrained DenseCap model by running the following script:

sh scripts/download_pretrained_model.sh

This will download a zipped version of the model (about 1.1 GB) to `data/models/densecap/densecap-pretrained-vgg16.t7.zip`, unpack it to `data/models/densecap/densecap-pretrained-vgg16.t7` (about 1.2 GB) and then delete the zipped version.





