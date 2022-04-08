import os
import sys

import numpy as np

from logging import error

from common import read_conll, process_sentences, load_ner_model
from common import load_viterbi_probabilities, viterbi_path
from common import encode, write_result
from common import argument_parser


def main(argv):
    argparser = argument_parser('predict')
    args = argparser.parse_args(argv[1:])

    ner_model, tokenizer, labels, config = load_ner_model(args.ner_model_dir)
    max_seq_len = config['max_seq_length']

    label_map = { t: i for i, t in enumerate(labels) }
    inv_label_map = { v: k for k, v in label_map.items() }

    if args.viterbi:
        try:
            init_prob, trans_prob = load_viterbi_probabilities(
                args.ner_model_dir, label_map)
        except Exception as e:
            error('failed to load viterbi probabilities: {}'.format(e))
            init_prob, trans_prob, args.viterbi = None, None, False

    test_words, dummy_labels = read_conll(args.test_data, mode='test')
    test_data = process_sentences(test_words, dummy_labels, tokenizer,
                                  max_seq_len)

    test_x = encode(test_data.combined_tokens, tokenizer, max_seq_len)

    probs = ner_model.predict(test_x, batch_size=args.batch_size)

    pred_labels = []
    label_probs = []
    if not args.viterbi:
        preds = np.argmax(probs, axis=-1)
        for i, (pred, prob) in enumerate(zip(preds, probs)):
            pred_labels.append([inv_label_map[t] for t in
                                pred[1:len(test_data.tokens[i])+1]])
            label_probs.append([prob[i+1][t] for i, t in
                                enumerate(pred[1:len(test_data.tokens[i])+1])])
    else:
        for i, prob in enumerate(probs):
            cond_prob = prob[1:len(test_data.tokens[i])+1]
            path = viterbi_path(init_prob, trans_prob, cond_prob)
            pred_labels.append([inv_label_map[i] for i in path])
            label_probs.append(['TODO' for i in path])    # not implemented

    if args.probabilities:
        pred_labels = [
            [f'{l}\t{p:.3f}' for l, p in zip(ls, ps)]
            for ls, ps in zip(pred_labels, label_probs)
        ]

    write_result(
        args.output_file, test_data.words, test_data.lengths,
        test_data.tokens, test_data.labels, pred_labels, mode='predict'
    )

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
