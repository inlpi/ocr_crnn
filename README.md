# ocr_crnn
Implementing a CRNN for OCR in PyTorch

This project from September 2020 implements a Convolutional Recurrent Neural Network (CRNN) for Optical Character Recognition (OCR).

# Data & Preprocessing

The data used in this project is from the [Visual Geometry Group](https://www.robots.ox.ac.uk/~vgg/data/text/) and is gathered and preprocessed with the scripts preprocessing.py and pad_labels.py. The resulting data used for training, validating and testing the model is in the respective directories. Some files are zipped (as one or multiple zip parts) to fit on this platform. This has been done and can be undone with 7zip for example.

# Model

The model and the code for training, validation and testing are part of network.py. The model is a CRNN consisting of a 7-layer CNN with several ReLU activations, pooling layers and batch normalization operations inbetween as well as a 2-layer bidirectional LSTM. The model outputs log probalities and for the loss we use the Connectionist temporal classification (CTC) function. Everything is implemented with PyTorch.

# References

Jaderberg, Max, et al. "Synthetic data and artificial neural networks for natural scene text recognition." arXiv preprint arXiv:1406.2227 (2014).

Jaderberg, Max, et al. "Reading text in the wild with convolutional neural networks." International journal of computer vision 116.1 (2016): 1-20.

Shi, Baoguang, Xiang Bai, and Cong Yao. "An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition." IEEE transactions on pattern analysis and machine intelligence 39.11 (2016): 2298-2304.
