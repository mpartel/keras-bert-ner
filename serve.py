from queue import Queue, SimpleQueue
import sys
from threading import Thread
import unicodedata

from flask import Flask, request
import waitress
import logging

import numpy as np
import tensorflow as tf

from common import process_sentences, load_ner_model
from common import encode, write_result
from common import argument_parser


DEFAULT_MODEL_DIR = 'ner-model'


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def tag():
    data = request.get_json() if request.method == 'POST' else request.values
    text = data['text']
    tokenized = data.get('tokenized') in ('1', 'True', 'true')
    return app.tagger_thread.tag(text, tokenized)


# We're having trouble using the model from multiple threads:
#   https://github.com/keras-team/keras/issues/11290
#   https://github.com/keras-team/keras/issues/5223
# Probably upgrading to TensorFlow 2 or something would fix it,
# but I don't know all the trouble that would entail, so we use a worker thread.
class TaggerThread(Thread):
    def __init__(self, model_dir):
        super().__init__(daemon=True)
        self.model_dir = model_dir
        self.queue = Queue(maxsize=2)

    def run(self):
        tagger = Tagger.load(self.model_dir)
        while True:
            command = self.queue.get()
            try:
                if command['type'] == 'ping':
                    command['callback']('pong')
                if command['type'] == 'tag':
                    text = command['text']
                    tokenized = command['tokenized']
                    result = tagger.tag(text, tokenized)
                    command['callback'](result)
            except BaseException as err:
                command['error_callback'](err)
            finally:
                self.queue.task_done()

    def ping(self):
        return self._send_command('ping', {})

    def tag(self, text, tokenized):
        return self._send_command('tag', {'text': text, 'tokenized': tokenized})

    def _send_command(self, type, args):
        response_queue = SimpleQueue()
        command = args.copy()
        command['type'] = type
        command['callback'] = lambda x: response_queue.put(['ok', x])
        command['error_callback'] = lambda e: response_queue.put(['err', e])
        self.queue.put(command)
        resp = response_queue.get()
        if resp[0] == 'ok':
            return resp[1]
        else:
            raise resp[1]

class Tagger(object):
    def __init__(self, session, graph, model, tokenizer, labels, config):
        self.session = session
        self.graph = graph
        self.model = model
        self.tokenizer = tokenizer
        self.labels = labels
        self.config = config

    def tag(self, text, tokenized=False):
        max_seq_len = self.config['max_seq_length']
        inv_label_map = { i: l for i, l in enumerate(self.labels) }
        if tokenized:
            words = text.split()    # whitespace tokenization
        else:
            words = tokenize(text)    # approximate BasicTokenizer
        dummy = ['O'] * len(words)
        data = process_sentences([words], [dummy], self.tokenizer, max_seq_len)
        x = encode(data.combined_tokens, self.tokenizer, max_seq_len)
        with self.session.as_default():
            with self.graph.as_default():
                probs = self.model.predict(x, batch_size=8)
        preds = np.argmax(probs, axis=-1)
        pred_labels = []
        for i, pred in enumerate(preds):
            pred_labels.append([inv_label_map[t]
                                for t in pred[1:len(data.tokens[i])+1]])
        lines = write_result(
            '/dev/null', data.words, data.lengths,
            data.tokens, data.labels, pred_labels, mode='predict'
        )
        return ''.join(lines)

    @classmethod
    def load(cls, model_dir):
        # session/graph for multithreading, see https://stackoverflow.com/a/54783311
        session = tf.Session()
        graph = tf.get_default_graph()
        with graph.as_default():
            with session.as_default():
                model, tokenizer, labels, config = load_ner_model(model_dir)
                tagger = cls(session, graph, model, tokenizer, labels, config)
        return tagger


punct_chars = set([
    chr(i) for i in range(sys.maxunicode)
    if (unicodedata.category(chr(i)).startswith('P') or
        ((i >= 33 and i <= 47) or (i >= 58 and i <= 64) or
         (i >= 91 and i <= 96) or (i >= 123 and i <= 126)))
])

translation_table = str.maketrans({ c: ' '+c+' ' for c in punct_chars })


def tokenize(text):
    return text.translate(translation_table).split()


def main(argv):
    argparser = argument_parser('serve')
    args = argparser.parse_args(argv[1:])
    if args.ner_model_dir is None:
        args.ner_model_dir = DEFAULT_MODEL_DIR
    app.tagger_thread = TaggerThread(args.ner_model_dir)
    app.tagger_thread.start()
    print("Loading model...")
    app.tagger_thread.ping()
    print("Model loaded. Starting server.")
    logging.getLogger('waitress').setLevel(logging.INFO)
    waitress.serve(app, host=args.host, port=int(args.port), threads=2)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
