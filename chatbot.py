"""
CITS4404 Group C1
Trains and enables user interaction with the chatbot

Code from:
https://github.com/chiphuyen/stanford-tensorflow-tutorials/tree/master/assignments/chatbot
https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/
"""
from __future__ import division
from __future__ import print_function

import argparse
import os
import random
import sys
import time

import numpy as np
import tensorflow as tf

from model import ChatBotModel
import config
import data_utils


def get_random_bucket(train_buckets_scale):
    """ Get a random bucket from which to choose a training sample """
    # random.seed(0)
    rand = random.random()
    return min([i for i in range(len(train_buckets_scale))
                if train_buckets_scale[i] > rand])


def assert_lengths(encoder_size, decoder_size, encoder_inputs, decoder_inputs, decoder_masks):
    """
    Assert that the encoder inputs, decoder inputs, and decoder masks are
    of the expected lengths
    """
    if len(encoder_inputs) != encoder_size:
        raise ValueError("Encoder length must be equal to the one in bucket,"
                         " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size:
        raise ValueError("Decoder length must be equal to the one in bucket,"
                         " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(decoder_masks) != decoder_size:
        raise ValueError("Weights length must be equal to the one in bucket,"
                         " %d != %d." % (len(decoder_masks), decoder_size))


def run_step(sess, model, encoder_inputs, decoder_inputs, decoder_masks, bucket_id, forward_only):
    """
    Run one step or pass in training
    @forward_only: boolean value to decide whether a backward path should be created

    forward_only is set to True when you just want to evaluate on the test set,
    or when you want to the bot to be in chat mode.
    """
    encoder_size, decoder_size = config.BUCKETS[bucket_id]
    assert_lengths(encoder_size, decoder_size, encoder_inputs, decoder_inputs, decoder_masks)

    input_feed = {}
    for step in range(encoder_size):
        input_feed[model.encoder_inputs[step].name] = encoder_inputs[step]
    for step in range(decoder_size):
        input_feed[model.decoder_inputs[step].name] = decoder_inputs[step]
        input_feed[model.decoder_masks[step].name] = decoder_masks[step]

    last_target = model.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([model.batch_size], dtype=np.int32)

    # output feed: depends on whether we do a backward step or not.
    if not forward_only:
        output_feed = [model.train_ops[bucket_id],  # update op that does SGD.
                       model.gradient_norms[bucket_id],  # gradient norm.
                       model.losses[bucket_id]]  # loss for this batch.
    else:
        output_feed = [model.losses[bucket_id]]  # loss for this batch.
        for step in range(decoder_size):  # output logits.
            output_feed.append(model.outputs[bucket_id][step])

    outputs = sess.run(output_feed, input_feed)
    if not forward_only:
        return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
    else:
        return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.


def get_buckets():
    """
    Load the dataset into buckets based on their lengths.
    """
    test_buckets = data_utils.load_data('test_ids.enc', 'test_ids.dec')
    data_buckets = data_utils.load_data('train_ids.enc', 'train_ids.dec')
    train_bucket_sizes = [len(data_buckets[b]) for b in range(len(config.BUCKETS))]
    print("Number of samples in each bucket:\n", train_bucket_sizes)
    train_total_size = sum(train_bucket_sizes)

    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in range(len(train_bucket_sizes))]
    print("Bucket scale:\n", train_buckets_scale)
    return test_buckets, data_buckets, train_buckets_scale


def get_skip_step(iteration):
    if iteration < 100:
        return 30
    return 100


def check_restore_parameters(sess, saver):
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(config.CPT_PATH + '/checkpoint'))
    if ckpt and ckpt.model_checkpoint_path:
        print("Loading parameters for the Chatbot")
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Initializing fresh parameters for the Chatbot")


def eval_test_set(sess, model, test_buckets, testing_loss_summary, file_writer):
    """ Evaluate on the test set. """
    for bucket_id in range(len(config.BUCKETS)):
        if len(test_buckets[bucket_id]) == 0:
            print("  Test: empty bucket %d" % (bucket_id))
            continue
        start = time.time()
        encoder_inputs, decoder_inputs, decoder_masks = data_utils.get_batch(test_buckets[bucket_id],
                                                                             bucket_id,
                                                                             batch_size=config.BATCH_SIZE)
        _, step_loss, _ = run_step(sess, model, encoder_inputs, decoder_inputs,
                                   decoder_masks, bucket_id, True)

        bucket_value = testing_loss_summary.value.add()
        bucket_value.tag = "testing_loss_bucket_%d" % bucket_id
        bucket_value.simple_value = step_loss
        file_writer.add_summary(testing_loss_summary, model.global_step.eval())

        print('Test bucket {}: loss {}, time {}'.format(bucket_id, step_loss, time.time() - start))

def train():
    """ Train the bot """
    test_buckets, data_buckets, train_buckets_scale = get_buckets()
    model = ChatBotModel(False, config.BATCH_SIZE)
    model.build_graph()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        print('Running session')
        sess.run(tf.global_variables_initializer())
        check_restore_parameters(sess, saver)

        iteration = model.global_step.eval()
        total_loss = 0

        file_writer = tf.summary.FileWriter(os.path.join(config.LOG_PATH, 'tensorboard'), sess.graph)
        training_loss_summary = tf.Summary()
        testing_loss_summary = tf.Summary()
        while True:
            skip_step = get_skip_step(iteration)
            bucket_id = get_random_bucket(train_buckets_scale)
            encoder_inputs, decoder_inputs, decoder_masks = data_utils.get_batch(data_buckets[bucket_id],
                                                                                 bucket_id,
                                                                                 batch_size=config.BATCH_SIZE)
            start = time.time()
            _, step_loss, _ = run_step(sess, model, encoder_inputs, decoder_inputs, decoder_masks, bucket_id, False)
            total_loss += step_loss
            iteration += 1

            if iteration % skip_step == 0:
                print('Iter {}: loss {}, time {}'.format(iteration, total_loss / skip_step, time.time() - start))

                bucket_value = training_loss_summary.value.add()
                bucket_value.tag = "training_loss_bucket_%d" % bucket_id
                bucket_value.simple_value = step_loss
                file_writer.add_summary(training_loss_summary, model.global_step.eval())

                start = time.time()
                total_loss = 0
                saver.save(sess, os.path.join(config.CPT_PATH, 'chatbot'), global_step=model.global_step)

                if iteration % (10 * skip_step) == 0:
                    # Run evals on development set and print their loss
                    eval_test_set(sess, model, test_buckets, testing_loss_summary, file_writer)
                    start = time.time()
                sys.stdout.flush()


def get_user_input():
    """ Get user's input, which will be transformed into encoder input later """
    print("> ", end="")
    sys.stdout.flush()
    return sys.stdin.readline()


def find_right_bucket(length):
    """ Find the proper bucket for an encoder input based on its length """
    return min([b for b in range(len(config.BUCKETS))
                if config.BUCKETS[b][0] >= length])


def construct_response(output_logits, inv_dec_vocab):
    """
    Construct a response to the user's encoder input.
    @output_logits: the outputs from sequence to sequence wrapper.
    output_logits is decoder_size np array, each of dim 1 x DEC_VOCAB

    This is a greedy decoder - outputs are just argmaxes of output_logits.
    """
    print(output_logits[0])
    outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
    # If there is an EOS symbol in outputs, cut them at that point.
    if config.EOS_ID in outputs:
        outputs = outputs[:outputs.index(config.EOS_ID)]
    # Print out sentence corresponding to outputs.
    return " ".join([tf.compat.as_str(inv_dec_vocab[output]) for output in outputs])


def chat():
    """ in test mode, we don't to create the backward path """
    _, enc_vocab = data_utils.load_vocab(os.path.join(config.PROCESSED_PATH, 'vocab.enc'))
    inv_dec_vocab, _ = data_utils.load_vocab(os.path.join(config.PROCESSED_PATH, 'vocab.dec'))

    model = ChatBotModel(True, batch_size=1)
    model.build_graph()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        check_restore_parameters(sess, saver)
        output_file = open('/Users/EleanorLeung/Documents/CITS4404/chatbot/output_convo.txt', 'a+')
        # Decode from standard input.
        max_length = config.BUCKETS[-1][0]
        print('Talk to me! Enter to exit. Max length is', max_length)
        while True:
            line = str.encode(get_user_input())
            if len(line) > 0 and line[-1] == '\n':
                line = line[:-1]
            if line == '':
                break
            output_file.write('HUMAN: ' + str(line) + '\n')
            token_ids = data_utils.sentence2id(enc_vocab, line)
            if len(token_ids) > max_length:
                print('Max length I can handle is:', max_length)
                line = get_user_input()
                continue
            bucket_id = find_right_bucket(len(token_ids))
            # Get a 1-element batch to feed the sentence to the model.
            encoder_inputs, decoder_inputs, decoder_masks = data_utils.get_batch([(token_ids, [])],
                                                                                 bucket_id,
                                                                                 batch_size=1)
            # Get output logits for the sentence.
            _, _, output_logits = run_step(sess, model, encoder_inputs, decoder_inputs,
                                           decoder_masks, bucket_id, True)
            response = construct_response(output_logits, inv_dec_vocab)
            print(response)
            output_file.write('BOT: ' + response + '\n')
        output_file.write('=============================================\n')
        output_file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices={'train', 'chat'},
                        default='train', help="mode. if not specified, it's in the train mode")
    args = parser.parse_args()

    if not os.path.isdir(config.PROCESSED_PATH):
        data_utils.prepare_raw_data()
        data_utils.process_data()
    print('Data ready!')
    data_utils.make_dir(config.CPT_PATH)

    if args.mode == 'train':
        train()
    elif args.mode == 'chat':
        chat()


if __name__ == '__main__':
    main()