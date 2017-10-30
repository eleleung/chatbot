"""
CITS4404 Group C1

App that starts the tensorflow session and processes the responses from the message request
"""

from flask import Flask, render_template, request
from flask import jsonify

import os.path

import tensorflow as tf

import data_utils
from model import ChatBotModel
import config
import chatbot

app = Flask(__name__,static_url_path="/static")

@app.route('/message', methods=['POST'])
def reply():
    message, response = get_response(request.form['msg'])
    return jsonify({'text': response})

@app.route("/")
def index():
    return render_template("index.html")


# dirty chatbot init
sess = tf.Session()
_, enc_vocab = data_utils.load_vocab(os.path.join(config.PROCESSED_PATH, 'vocab.enc'))
inv_dec_vocab, _ = data_utils.load_vocab(os.path.join(config.PROCESSED_PATH, 'vocab.dec'))

model = ChatBotModel(True, batch_size=1)
model.build_graph()

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
chatbot.check_restore_parameters(sess, saver)
max_length = config.BUCKETS[-1][0]

output_file = open(os.path.join(config.PROCESSED_PATH, config.OUTPUT_FILE), 'a+')

def get_response(message):
    line = str.encode(message)
    if len(line) > 0 and line[-1] == '\n':
        line = line[:-1]
    if line == '':
        response = 'What did you say?'
        output_file.write('Human: ' + message + '\n' + 'Bot: ' + str(response) + '\n')
        return message, response

    token_ids = data_utils.sentence2id(enc_vocab, line)
    if len(token_ids) > max_length:
        response = ('The maximum length I can handle is ', max_length)
        output_file.write('Human: ' + message + '\n' + 'Bot: ' + str(response) + '\n')
        return message, response

    bucket_id = chatbot.find_right_bucket(len(token_ids))
    encoder_inputs, decoder_inputs, decoder_masks = data_utils.get_batch([(token_ids, [])],
                                                                         bucket_id,
                                                                         batch_size=1)
    _, _, output_logits = chatbot.run_step(sess, model, encoder_inputs, decoder_inputs,
                                           decoder_masks, bucket_id, True)
    response = chatbot.construct_response(output_logits, inv_dec_vocab)
    output_file.write('Human: ' + message + '\n' + 'Bot: ' + str(response) + '\n')

    return message, response


if __name__ == "__main__":
    app.run(port=5000)
