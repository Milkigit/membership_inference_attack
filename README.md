# Overview
* Run the preprocess.py file to obtain the normalized dataset. The normalized data is in csv format, with the first column represent label and the remaining columns represent features.
* Run the main.py file to start the program, one should specify name for normalized dataset when run this program.  The main.py file would revoke the experiment.py file to conduct the experiment, which would then revoke other files to implement specific functions.

# Functionality of Files
* preprocess.py: contains functions to proprocess the original dataset.
* main.py: the program entry.
* experiment.py: contains the main procedures for the experiment.
* attack.py: contains three functions to train the target model, shadow models and attack models, respectively.
* classifier.py: contains functions to construct and train different classifiers.
* neural_network.py: contain functions to reformat the image dataset and construct the convolutional neural network.
* config.py: define some global constants.