# CITS4404 Project
## Neural Networks - Team C1

This folder contains all the source code involved in training and testing a chatbot
using a seq2seq model from the TensorFlow API. Any file or code that has been adapted from or written 
by another author has been explicitly marked at the top of the file. To breakdown each folder:

/data - all the raw data can be found in this folder

/database_daos - database access objects for storing in MongoDB

/friends_corpus - contains utilities to process the raw Friends script. The final cleaned corpus is located 
in /friends_corpus/friends_data

/output_convo - snippets of interaction with the model from the 2 experiments described in the report.
**Warning that experiment_2.txt has bad language**

/processed_data - the results of pre-processing the raw data

/twitter_scraper - script that hits the Twitter API and continuously streams for Twitter data.
Run this if you would like to collect more Twitter data.

/ui - contains all the code required to run the web interface to interact with a chatbot model

chatbot.py - trains and enables user interaction with the chatbot

config.py - states the config parameters and hyperparameters

data_utils - pre-processes the data that is to be fed into the model

model.py - the structure and architecture of the model

### System Requirements
You will need to install Python 3 and Tensorflow to be able to run this

### How to run the chatbot
You will have to train the chatbot from scratch as checkpoints haven't been included
in the submission due to their large size (checkpoints were over 1gb in size)

To train the model from the raw data provided:

1. Clone the repo
2. In config.py, update the folder paths (replace '/Users/EleanorLeung/Documents/CITS4404/')
3. Run data_utils.py to prepare the model for the data. This will take some time (~30 minutes)and the results will be placed in /processed_data
4. Run chatbot.py with --mode=train to start training the model. This will create a new folder /models that will contain
a folder with checkpoints (/checkpoints) that saves the state of your model after a certain number of iterations.
It will also create a log folder (/log) that contains TensorBoard summaries that you can visualise.
5. Stop the training at your discretion (usually when the loss has stabilised)
6. To test the model, run chatbot.py with --mode=chat to interact with it.
7. If you would like to use the web interface, go into /ui and run app.py
8. Go to localhost:5000