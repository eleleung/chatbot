"""
Config parameters and hyperparameters
"""

# cornell movie-dialogs data paths
DATA_PATH = '/Users/EleanorLeung/Documents/CITS4404/chatbot/data/cornell movie subset'
CONVO_FILE = 'movie_conversations.txt'
LINE_FILE = 'movie_lines.txt'
OUTPUT_FILE = 'output_convo.txt'
PROCESSED_PATH = 'processed_data'
CPT_PATH = 'checkpoints'

# min. num of times a word must appear to be added to vocab
THRESHOLD = 1

# reserved vocab tokens
PAD_ID = 0
UNK_ID = 1
START_ID = 2
EOS_ID = 3

# percentage split of data between training and testing
TEST_SET_PERCENTAGE = 0

# bucket sizes
BUCKETS = [(8, 10), (12, 14), (16, 19)]

# model parameters
NUM_LAYERS = 3
HIDDEN_SIZE = 256
BATCH_SIZE = 64

LR = 0.5
MAX_GRAD_NORM = 5.0

NUM_SAMPLES = 512
