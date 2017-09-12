#!/usr/bin/env python
__author__ = 'jesse'

import argparse
import os
import pickle
from functions import fit_classifier, get_margin_kappa, get_labels


def main():

    nb_objects = 32

    # Convert flags to local variables.
    indir = FLAGS_indir
    kernel = FLAGS_kernel
    contexts = [pair.split(',') for pair in FLAGS_contexts.split('.')]
    test_oidx = FLAGS_test_oidx
    pidx = FLAGS_pidx
    outfile = FLAGS_outfile

    labels = get_labels(indir, os.path.join(indir, 'full_annotations.pickle'))
    feature_fn = os.path.join(indir, 'features.pickle')
    with open(feature_fn, 'rb') as f:
        object_feats = pickle.load(f)

    # Fit SVMs.
    train_pairs = [(oidx, labels[oidx][pidx])
                   for oidx in [oidx for oidx in range(nb_objects) if oidx != test_oidx]
                   if labels[oidx][pidx] == 0 or labels[oidx][pidx] == 1]
    cs = {}
    kas = {}
    for b, m in contexts:
        if b not in cs:
            cs[b] = {}
            kas[b] = {}
        c = fit_classifier(kernel, b, m, train_pairs, object_feats)
        ka = get_margin_kappa(c, b, m, train_pairs, object_feats,
                              xval=train_pairs, kernel=kernel)
        cs[b][m] = c
        kas[b][m] = ka

    # Write outfile.
    with open(outfile, 'wb') as f:
        pickle.dump([cs, kas], f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True,
                        help="data directory")
    parser.add_argument('--kernel', type=str, required=True,
                        help="SVM kernel to use (linear, poly, rbf)")
    parser.add_argument('--contexts', type=str, required=True,
                        help="period separated list of contexts that are b,m comma separated")
    parser.add_argument('--test_oidx', type=int, required=True,
                        help="test object to hold out")
    parser.add_argument('--pidx', type=int, required=True,
                        help="predicate to work with")
    parser.add_argument('--outfile', type=str, required=True,
                        help="pickle file to write out")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
