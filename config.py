"""
CITS4404 Group C1
Config parameters and hyperparameters
"""

# cornell movie-dialogs data paths
CORNELL_DATA_PATH = '/Users/EleanorLeung/Documents/CITS4404/chatbot/data/cornell movie-dialogs corpus'
CORNELL_CONVO_FILE = 'movie_conversations.txt'
CORNELL_LINE_FILE = 'movie_lines.txt'

# twitter dataset path
TWITTER_DATA_PATH = '/Users/EleanorLeung/Documents/CITS4404/chatbot/twitter_scraper/twitter_data/'
TWITTER_CONVO_FILE = 'cleaned_corpus.txt'

# friends dataset path
FRIENDS_DATA_PATH = '/Users/EleanorLeung/Documents/CITS4404/chatbot/friends_corpus/friends_data/'
FRIENDS_RAW_DATA = 'friends-final.txt'
FRIENDS_CONVO_FILE = 'cleaned_corpus.txt'

OUTPUT_FILE = 'output_convo.txt'
PROCESSED_PATH = '/Users/EleanorLeung/Documents/CITS4404/chatbot/processed_data/'
CPT_PATH = '/Users/EleanorLeung/Documents/CITS4404/chatbot/models/checkpoints'
LOG_PATH = '/Users/EleanorLeung/Documents/CITS4404/chatbot/models/logs'

# min. num of times a word must appear to be added to vocab
THRESHOLD = 2

# reserved vocab tokens
PAD_ID = 0
UNK_ID = 1
START_ID = 2
EOS_ID = 3

# percentage split of data between training and testing
TEST_SET_PERCENTAGE = 0.3

# bucket sizes
BUCKETS = [(8, 10), (12, 14), (16, 19)]

# model parameters
NUM_LAYERS = 3
HIDDEN_SIZE = 256
BATCH_SIZE = 64

LR = 0.5
MAX_GRAD_NORM = 5.0
DROPOUT = 0.2

NUM_SAMPLES = 512

# experiment_2
TESTSET_SIZE = 169410
ENC_VOCAB = 34247
DEC_VOCAB = 32277