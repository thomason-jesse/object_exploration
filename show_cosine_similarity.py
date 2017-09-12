#!/usr/bin/env python
__author__ = 'jesse'

import argparse
import gensim
import numpy as np
import os
import pickle
from functions import get_labels


def main():

    # Convert flags to local variables.
    indir = FLAGS_indir
    word_embeddings_fn = FLAGS_word_embeddings
    required_examples = FLAGS_required_examples
    train_kappa_threshold = FLAGS_train_kappa_threshold
    alternative_labels = FLAGS_alternative_labels
    max_cos = FLAGS_max_cos
    top_k = FLAGS_top_k
    if top_k is None:
        top_k = 1

    labels = get_labels(indir, alternative_labels)

    f_train = [10, 3, 27, 7] + [18, 2, 20, 17] + [5, 14, 8, 15] + [1, 30, 29, 31]
    f_test = [21, 24, 19, 23] + [16, 0, 4, 9] + [22, 28, 12, 25] + [11, 6, 26, 13]

    with open(os.path.join(indir, 'predicates.pickle'), 'rb') as f:
        predicates = pickle.load(f)
        nb_predicates = len(predicates)
    # predicates = ['red','green','blue','light','medium','heavy','marbles','screws','beans','rice']  # DEBUG
    # nb_predicates = len(predicates)  # DEBUG

    # Pre-calculate matrix of cosine similarity of word embeddings.
    wvb = True if word_embeddings_fn.split('.')[-1] == 'bin' else False
    wv = gensim.models.KeyedVectors.load_word2vec_format(word_embeddings_fn, binary=wvb,
                                                         limit=150000)

    preds_above = []
    if max_cos is None:
        print ','.join([predicates[pidx] for pidx in range(nb_predicates)])
    for test_pidx in range(nb_predicates):

        test_oidxs = [oidx for oidx in f_test if labels[oidx][test_pidx] == 0 or labels[oidx][test_pidx] == 1]
        train_oidxs = [oidx for oidx in f_train if oidx not in test_oidxs]

        valid_predicates = [pidx for pidx in range(nb_predicates)
                            if sum([1 if labels[oidx][pidx] == 1 else 0
                                    for oidx in train_oidxs]) >= required_examples
                            and sum([1 if labels[oidx][pidx] == 0 else 0
                                     for oidx in train_oidxs]) >= required_examples]
        train_threshold_preds = [pidx for pidx in range(nb_predicates)
                                 if sum([1 if labels[oidx][pidx] == 1 or labels[oidx][pidx] == 0 else 0
                                         for oidx in train_oidxs]) >= train_kappa_threshold]

        trainable_preds = [pidx for pidx in range(nb_predicates)
                           if pidx in valid_predicates and pidx in train_threshold_preds
                           and pidx != test_pidx]

        if predicates[test_pidx] in wv.vocab:
            pred_cosine = [(1 + wv.similarity(predicates[test_pidx], predicates[pjdx])) / 2.0
                           if predicates[pjdx] in wv.vocab else 0 for pjdx in range(nb_predicates)]
        else:
            pred_cosine = [0 if pjdx != test_pidx else 1 for pjdx in range(nb_predicates)]
        if max_cos is None:
            print ','.join([str(pc) for pc in pred_cosine])
        max_sims = []
        test_max_cos = []
        while len(max_sims) < top_k:
            test_max_cos.append(np.max([pred_cosine[pjdx] for pjdx in trainable_preds
                                        if pjdx not in max_sims]))
            max_sims.extend([i for i, x in enumerate(pred_cosine)
                             if np.isclose(x, test_max_cos[-1])])
        if max_cos is not None and test_max_cos[-1] >= max_cos:  # DEBUG
            print ("'" + str(predicates[test_pidx]) + "': " + ','.join([predicates[midx] for midx in max_sims]) +
                   " (" + str(test_max_cos) + ")")
            preds_above.append(test_pidx)
    if max_cos is not None:
        print ','.join([predicates[pidx] for pidx in preds_above])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True,
                        help="data directory")
    parser.add_argument('--word_embeddings', type=str, required=True,
                        help="word embeddings binary to use")
    parser.add_argument('--required_examples', type=int, required=True,
                        help="how many positive and negative examples per predicate to qualify")
    parser.add_argument('--train_kappa_threshold', type=int, required=True,
                        help="number of examples required before trusting a training predicate classifier")
    parser.add_argument('--alternative_labels', type=str, required=False,
                        help="specify labels pickle; labels in this pickle will override defaults")
    parser.add_argument('--max_cos', type=float, required=False,
                        help="show the max cosine neighbor(s) of a predicate if the similarity exceeds this")
    parser.add_argument('--top_k', type=int, required=False,
                        help="if tied max don't reach k preds, get next highest")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
