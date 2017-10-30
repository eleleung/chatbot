"""
Pre-processes data to be fed into the model

Code from:
https://github.com/chiphuyen/stanford-tensorflow-tutorials/tree/master/assignments/chatbot
https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/data_utils.py
"""
from __future__ import print_function

import os
import random
import re as regex

import numpy as np

import config
from database_daos import phrase_pair_dao as phrase_pair


def prepare_raw_data():
    print('Splitting raw data into train set and test set...')
    id2line = get_lines()
    convos = get_convos()
    questions, answers = question_answers(id2line, convos)
    prepare_dataset(questions, answers, config.TEST_SET_PERCENTAGE)

def process_data():
    print('Preparing data for model...')
    build_vocab('train.enc')
    build_vocab('train.dec')

    token2id('train', 'enc')
    token2id('train', 'dec')
    token2id('test', 'enc')
    token2id('test', 'dec')

def get_lines():
    """ Read all movie lines from movie_lines.txt """
    id2line = {}
    file_path = os.path.join(config.CORNELL_DATA_PATH, config.CORNELL_LINE_FILE)
    with open(file_path, 'rb') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.split(b' +++$+++ ')
            if len(parts) == 5:
                if parts[4][-1] == '\n':
                    parts[4] = parts[4][:-1]
                id2line[parts[0]] = parts[4]
    return id2line

def get_convos():
    """ Get conversation structure from the raw data movie_conversations.txt """
    file_path = os.path.join(config.CORNELL_DATA_PATH, config.CORNELL_CONVO_FILE)
    convos = []
    with open(file_path, 'rb') as f:
        for line in f.readlines():
            parts = line.split(b' +++$+++ ')
            if len(parts) == 4:
                convo = []
                for line in parts[3][1:-2].split(b', '):
                    convo.append(line[1:-1])
                convos.append(convo)

    return convos

def question_answers(id2line, convos):
    """ Divide the dataset into two sets: questions and answers. """
    questions, answers = [], []
    for convo in convos:
        for index, line in enumerate(convo[:-1]):
            questions.append(id2line[convo[index]])
            answers.append(id2line[convo[index + 1]])
    assert len(questions) == len(answers)
    return questions, answers

def prepare_dataset(questions, answers, data_split=0.7):
    assert len(questions) == len(answers)

    phrase_pairs = list()
    for i in range(len(questions)):
        phrase_pairs.append(phrase_pair.PhrasePair(name="Cornell Movie Dialog",
                                                   x=questions[i],
                                                   y=answers[i]))
    phrase_pair.save_data(phrase_pairs)

def prepare_model(data_split=0.7):
    phrase_pairs = phrase_pair.load_data()

    # split into training and testing
    # random convos to create the test set - random indexes from questions[]
    random.seed(0)
    cut_off = int(data_split * len(phrase_pairs))
    with open('config.py', 'ab') as cf:
        cf.write(b'\n' + b'TESTSET_SIZE = ' + str.encode(str(cut_off)))
    test_ids = random.sample([i for i in range(len(phrase_pairs))], cut_off)

    training_enc = list()
    training_dec = list()
    testing_enc = list()
    testing_dec = list()

    for index in range(len(phrase_pairs)):
        if index in test_ids:
            testing_enc.append(phrase_pairs[index].dict['x'])
            testing_dec.append(phrase_pairs[index].dict['y'])
        else:
            training_enc.append(phrase_pairs[index].dict['x'])
            training_dec.append(phrase_pairs[index].dict['y'])

    # build enc and dec vocab of training set
    vocab_enc = {}
    for line in training_enc:
        for token in basic_tokenizer(line):
            if token not in vocab_enc:
                vocab_enc[token] = 0
            vocab_enc[token] += 1

    vocab_dec = {}
    for line in training_dec:
        for token in basic_tokenizer(line):
            if token not in vocab_dec:
                vocab_dec[token] = 0
            vocab_dec[token] += 1

    sorted_vocab_enc = sorted(vocab_enc, key=vocab_enc.get, reverse=True)
    sorted_vocab_dec = sorted(vocab_dec, key=vocab_dec.get, reverse=True)

    vocab_enc_list = list()
    vocab_enc_list.append(b'<pad>')
    vocab_enc_list.append(b'<unk>')
    vocab_enc_list.append(b'<s>')
    vocab_enc_list.append(b'<\s>')
    index = 4

    for word in sorted_vocab_enc:
        if vocab_enc[word] < config.THRESHOLD:
            with open('config.py', 'ab') as cf:
                    cf.write(b'ENC_VOCAB = ' + str.encode(str(index)) + b'\n')
            break
        vocab_enc_list.append(word)
        index += 1

    vocab_enc_dict = {}
    for i in range(len(vocab_enc_list)):
        vocab_enc_dict.update({vocab_enc_list[i], i})

    # repeat for vocab_dec
    vocab_dec_dict = {}

    # token2id

    for line in training_enc:
        ids = []
        ids.extend(sentence2id(vocab_enc_dict, line))
        test_out_file.write(str.encode(' '.join(str(id_) for id_ in ids)) + b'\n')

    for line in training_dec:
        ids = []
        ids.extend(sentence2id(vocab_dec_dict, line))

    for line in testing_enc:
        ids = []
        ids.extend(sentence2id(vocab_enc_dict, line))

    for line in testing_dec:
        ids = []
        ids.extend(sentence2id(vocab_dec_dict, line))




def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass

def basic_tokenizer(line, normalise_digits=True):
    """ A basic tokenizer to tokenize text into tokens. """
    line = regex.sub(b'<u>', b'', line)
    line = regex.sub(b'</u>', b'', line)
    line = regex.sub(b'\[', b'', line)
    line = regex.sub(b'\]', b'', line)
    words = []
    _WORD_SPLIT = regex.compile(b"([.,!?\"'-<>:;)(])")
    _DIGIT_RE = regex.compile(b"\d")
    for fragment in line.strip().lower().split():
        for token in regex.split(_WORD_SPLIT, fragment):
            if not token:
                continue
            if normalise_digits:
                token = regex.sub(_DIGIT_RE, b'#', token)
            words.append(token)
    return words

def build_vocab(filename, normalize_digits=True):
    in_path = os.path.join(config.PROCESSED_PATH, filename)
    out_path = os.path.join(config.PROCESSED_PATH, 'vocab.{}'.format(filename[-3:]))

    vocab = {}
    test_vocab = {}

    with open(in_path, 'rb') as f:
        for line in f.readlines():
            for token in basic_tokenizer(line):
                if not token in vocab:
                    vocab[token] = 0
                vocab[token] += 1

    sorted_vocab = sorted(vocab, key=vocab.get, reverse=True)
    with open(out_path, 'wb') as f:
        f.write(b'<pad>' + b'\n')
        f.write(b'<unk>' + b'\n')
        f.write(b'<s>' + b'\n')
        f.write(b'<\s>' + b'\n')
        index = 4
        for word in sorted_vocab:
            if vocab[word] < config.THRESHOLD:
                with open('config.py', 'ab') as cf:
                    if filename[-3:] == 'enc':
                        cf.write(b'ENC_VOCAB = ' + str.encode(str(index)) + b'\n')
                    else:
                        cf.write(b'DEC_VOCAB = ' + str.encode(str(index)) + b'\n')
                break
            f.write(word + b'\n')
            index += 1

def load_vocab(vocab_path):
    with open(vocab_path, 'rb') as f:
        words = f.read().splitlines()
    return words, {words[i]: i for i in range(len(words))}

def sentence2id(vocab, line):
    return [vocab.get(token, vocab[b'<unk>']) for token in basic_tokenizer(line)]

def token2id(data, mode):
    """ Convert all the tokens in the data into their corresponding
    index in the vocabulary. """

    vocab_path = 'vocab.' + mode
    in_path = data + '.' + mode
    out_path = data + '_ids.' + mode

    _, vocab = load_vocab(os.path.join(config.PROCESSED_PATH, vocab_path))
    in_file = open(os.path.join(config.PROCESSED_PATH, in_path), 'rb')
    out_file = open(os.path.join(config.PROCESSED_PATH, out_path), 'wb')

    lines = in_file.read().splitlines()
    for line in lines:
        if mode == 'dec':  # we only care about '<s>' and </s> in encoder
            ids = [vocab[b'<s>']]
        else:
            ids = []
        ids.extend(sentence2id(vocab, line))
        if mode == 'dec':
            ids.append(vocab[b'<\s>'])
        out_file.write(str.encode(' '.join(str(id_) for id_ in ids)) + b'\n')

    # using db
    test_path = data + '_test_ids.' + mode
    test_out_file = open(os.path.join(config.PROCESSED_PATH, test_path), 'wb')

    convos = phrase_pair.load_data()
    for line in convos:
        if mode == 'dec':  # we only care about '<s>' and </s> in encoder
            ids = [vocab[b'<s>']]
        else:
            ids = []
        ids.extend(sentence2id(vocab, line))
        if mode == 'dec':
            ids.append(vocab[b'<\s>'])
        test_out_file.write(str.encode(' '.join(str(id_) for id_ in ids)) + b'\n')


def load_data(enc_filename, dec_filename, max_training_size=None):
    encode_file = open(os.path.join(config.PROCESSED_PATH, enc_filename), 'rb')
    decode_file = open(os.path.join(config.PROCESSED_PATH, dec_filename), 'rb')
    encode, decode = encode_file.readline(), decode_file.readline()
    data_buckets = [[] for _ in config.BUCKETS]
    i = 0
    while encode and decode:
        if (i + 1) % 10000 == 0:
            print("Bucketing conversation number", i)
        encode_ids = [int(id_) for id_ in encode.split()]
        decode_ids = [int(id_) for id_ in decode.split()]
        for bucket_id, (encode_max_size, decode_max_size) in enumerate(config.BUCKETS):
            if len(encode_ids) <= encode_max_size and len(decode_ids) <= decode_max_size:
                data_buckets[bucket_id].append([encode_ids, decode_ids])
                break
        encode, decode = encode_file.readline(), decode_file.readline()
        i += 1
    return data_buckets

def _pad_input(input_, size):
    return input_ + [config.PAD_ID] * (size - len(input_))

def _reshape_batch(inputs, size, batch_size):
    """ Create batch-major inputs. Batch inputs are just re-indexed inputs """
    batch_inputs = []
    for length_id in range(size):
        batch_inputs.append(np.array([inputs[batch_id][length_id]
                                    for batch_id in range(batch_size)], dtype=np.int32))
    return batch_inputs


def get_batch(data_bucket, bucket_id, batch_size=1):
    """ Return one batch to feed into the model """
    # only pad to the max length of the bucket
    encoder_size, decoder_size = config.BUCKETS[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    for _ in range(batch_size):
        encoder_input, decoder_input = random.choice(data_bucket)
        # pad both encoder and decoder, reverse the encoder
        encoder_inputs.append(list(reversed(_pad_input(encoder_input, encoder_size))))
        decoder_inputs.append(_pad_input(decoder_input, decoder_size))

    # now we create batch-major vectors from the data selected above.
    batch_encoder_inputs = _reshape_batch(encoder_inputs, encoder_size, batch_size)
    batch_decoder_inputs = _reshape_batch(decoder_inputs, decoder_size, batch_size)

    # create decoder_masks to be 0 for decoders that are padding.
    batch_masks = []
    for length_id in range(decoder_size):
        batch_mask = np.ones(batch_size, dtype=np.float32)
        for batch_id in range(batch_size):
            # we set mask to 0 if the corresponding target is a PAD symbol.
            # the corresponding decoder is decoder_input shifted by 1 forward.
            if length_id < decoder_size - 1:
                target = decoder_inputs[batch_id][length_id + 1]
            if length_id == decoder_size - 1 or target == config.PAD_ID:
                batch_mask[batch_id] = 0.0
        batch_masks.append(batch_mask)
    return batch_encoder_inputs, batch_decoder_inputs, batch_masks

if __name__ == '__main__':
    prepare_raw_data()
    process_data()