#!/usr/bin/env python
__author__ = 'jesse'
''' Train mult-modal mahalanobis-based neighbor classifiers.
'''

import argparse
import os


def get_decisions_from_weights(ds, ws, contexts):
    r = {}
    for pidx in ds:
        r[pidx] = sum([ds[pidx][b][m] * ws[pidx][contexts.index((b, m))]
                       for b, m in contexts]) / float(len(contexts))
    return r


def main():

    nb_objects = 32

    # Convert flags to local variables.
    indir = FLAGS_indir
    sparse_contexts = FLAGS_sparse_contexts
    kernel = FLAGS_kernel
    word_embeddings_fn = FLAGS_word_embeddings
    outdir = FLAGS_outdir
    retrain_svms = FLAGS_retrain_svms

    for oidx in range(nb_objects):
        cmd = ("perform_exploration_exp.py" +
               " --indir " + indir +
               " --test_oidx " + str(oidx) +
               " --sparse_contexts " + str(sparse_contexts) +
               " --kernel " + kernel +
               " --word_embeddings " + word_embeddings_fn +
               " --outdir " + outdir)
        if retrain_svms is not None:
            cmd += " --retrain_svms " + str(retrain_svms)
        print cmd
        os.system("condorify_cpu " + cmd + " " + str(oidx) + ".log")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True,
                        help="data directory")
    parser.add_argument('--sparse_contexts', type=int, required=True,
                        help="whether to enforce sparse contexts in confidences")
    parser.add_argument('--kernel', type=str, required=True,
                        help="SVM kernel to use (linear, poly, rbf)")
    parser.add_argument('--word_embeddings', type=str, required=True,
                        help="word embeddings binary to use")
    parser.add_argument('--outdir', type=str, required=True,
                        help="directory to write SVMs, confidences, and results pickles")
    parser.add_argument('--retrain_svms', type=int, required=False,
                        help="whether to retrain svms instead of loading from file")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
