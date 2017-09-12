#!/usr/bin/env python
__author__ = 'jesse'

import argparse
import os
import pickle
from functions import get_labels


def main():

    nb_objects = 32

    # Convert flags to local variables.
    indir = FLAGS_indir
    kernel = FLAGS_kernel
    word_embeddings_fn = FLAGS_word_embeddings
    required_examples = FLAGS_required_examples
    train_kappa_threshold = FLAGS_train_kappa_threshold
    outdir = FLAGS_outdir
    resample_test = FLAGS_resample_test
    alternative_labels = FLAGS_alternative_labels
    uniform_costs = FLAGS_uniform_costs
    degenerate = FLAGS_degenerate
    use_behavior_decisions = FLAGS_use_behavior_decisions
    assert required_examples >= 2
    assert 0 <= train_kappa_threshold

    print "reading in folds, labels, predicates, and features..."
    labels = get_labels(indir, alternative_labels)
    with open(os.path.join(indir, 'predicates.pickle'), 'rb') as f:
        predicates = pickle.load(f)
        nb_predicates = len(predicates)
    print "... done"

    # fixed splits of fully-annotated folds
    f_train = [10, 3, 27, 7] + [18, 2, 20, 17] + [5, 14, 8, 15] + [1, 30, 29, 31]
    f_test = [21, 24, 19, 23] + [16, 0, 4, 9] + [22, 28, 12, 25] + [11, 6, 26, 13]

    # DEBUG - imbalanced train 24 / test 8
    # f_train = [10, 3, 27, 7] + [18, 2, 20, 17] + [5, 14, 8, 15] + [1, 30, 29, 31] + [21, 24, 19, 23] + [16, 0, 4, 9]
    # f_test = [22, 28, 12, 25] + [11, 6, 26, 13]

    print "launching jobs..."
    for pidx in range(nb_predicates):
        oidxs = [oidx for oidx in f_test if labels[oidx][pidx] == 0 or labels[oidx][pidx] == 1]
        numpos = len([labels[oidx][pidx] for oidx in oidxs if labels[oidx][pidx] == 1])
        numneg = len([labels[oidx][pidx] for oidx in oidxs if labels[oidx][pidx] == 0])
        if numpos >= required_examples and numneg >= required_examples:
            cmd = ("leave_one_out_predicate_exp.py" +
                   " --indir " + indir +
                   " --test_pidx " + str(pidx) +
                   " --kernel " + kernel +
                   " --word_embeddings " + word_embeddings_fn +
                   " --required_examples " + str(required_examples) +
                   " --train_kappa_threshold " + str(train_kappa_threshold) +
                   " --resample_test " + str(resample_test) +
                   " --outfile " + os.path.join(outdir, str(pidx) + ".pickle"))
            if alternative_labels is not None:
                cmd += " --alternative_labels " + alternative_labels
            if uniform_costs is not None:
                cmd += " --uniform_costs " + str(uniform_costs)
            if degenerate is not None:
                cmd += " --degenerate " + str(degenerate)
            if use_behavior_decisions is not None:
                cmd += " --use_behavior_decisions " + str(use_behavior_decisions)
            print "... $ " + cmd
            os.system("condorify_cpu " + cmd + " " + str(pidx) + ".log")
        else:
            print "... '" + str(predicates[pidx]) + "' has too few pos,neg examples " + str((numpos, numneg))
    print "... done"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True,
                        help="data directory")
    parser.add_argument('--kernel', type=str, required=True,
                        help="SVM kernel to use (linear, poly, rbf)")
    parser.add_argument('--word_embeddings', type=str, required=True,
                        help="word embeddings binary to use")
    parser.add_argument('--required_examples', type=int, required=True,
                        help="how many positive and negative examples per predicate to qualify")
    parser.add_argument('--train_kappa_threshold', type=int, required=True,
                        help="number of examples required before trusting a training predicate classifier")
    parser.add_argument('--outdir', type=str, required=True,
                        help="directory to write results pickles")
    parser.add_argument('--resample_test', type=int, required=True,
                        help="whether to resample at test time and cut off exploration early")
    parser.add_argument('--alternative_labels', type=str, required=False,
                        help="specify labels pickle; labels in this pickle will override defaults")
    parser.add_argument('--uniform_costs', type=int, required=False,
                        help="whether to use a uniform cost function")
    parser.add_argument('--degenerate', type=int, required=False,
                        help="whether to use a degenerate transition graph")
    parser.add_argument('--use_behavior_decisions', type=int, required=False,
                        help="whether to use behavior-level kappas and decisions")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
