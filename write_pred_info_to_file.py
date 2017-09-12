#!/usr/bin/env python
__author__ = 'jesse'

import argparse
import gensim
import numpy as np
import operator
import os
import pickle
import random
from functions import get_data_for_classifier, fit_classifier, get_margin_kappa


def main():

    behaviors = ["drop", "grasp", "hold", "lift", "look", "lower", "press", "push"]
    modalities = ["audio_ispy", "color", "fpfh", "haptic_ispy"]  # "finger",
    nb_objects = 32

    # Convert flags to local variables.
    indir = FLAGS_indir
    kernel = FLAGS_kernel
    word_embeddings_fn = FLAGS_word_embeddings
    outdir = FLAGS_outdir

    print "reading in folds, labels, predicates, and features..."
    with open(os.path.join(indir, 'labels.pickle'), 'rb') as f:
        labels = pickle.load(f)
    with open(os.path.join(indir, 'predicates.pickle'), 'rb') as f:
        predicates = pickle.load(f)
        nb_predicates = len(predicates)
    gmm_feature_fn = os.path.join(indir, 'gmm_features.pickle')
    if os.path.isfile(gmm_feature_fn):
        with open(gmm_feature_fn, 'rb') as f:
            object_feats = pickle.load(f)
    else:
        print "... extracting normalized ispy features (removes unary features)"
        object_feats = {}
        for oidx in range(nb_objects):
            print "...... extracting from object " + str(oidx)
            with open(os.path.join(indir, str(oidx) + '.pickle'), 'rb') as f:
                d = pickle.load(f)
                object_feats[oidx] = {}
                for b in behaviors:
                    object_feats[oidx][b] = {}
                    for m in modalities:
                        if m in d[b].keys():
                            object_feats[oidx][b][m] = d[b][m]
        print "...... done"
        print "writing normalized features to file..."
        with open(gmm_feature_fn, 'wb') as f:
            pickle.dump(object_feats, f)
        print "... done"
    contexts = []
    for oidx in object_feats:
        contexts = []
        for b in object_feats[oidx]:
            for m in object_feats[oidx][b]:
                contexts.append((b, m))
                for tidx in range(len(object_feats[oidx][b][m])):
                    object_feats[oidx][b][m][tidx] = object_feats[oidx][b][m][tidx][0]  # one timestep eaach
    nb_contexts = len(contexts)
    valid_predicates = [pidx for pidx in range(nb_predicates)
                        if sum([1 if labels[oidx][pidx] == 1 else 0
                                for oidx in range(nb_objects)]) >= 1
                        and sum([1 if labels[oidx][pidx] == 0 else 0
                                 for oidx in range(nb_objects)]) >= 1]
    with open(os.path.join(indir, 'behavior_annotations.pickle'), 'rb') as f:
        behavior_annotations = pickle.load(f)
    with open(os.path.join(indir, 'modality_norms.pickle'), 'rb') as f:
        modality_annotations = pickle.load(f)
    print "... done"

    # Pre-calculate matrix of cosine similarity of word embeddings.
    print "pre-calculating word embeddings similarities..."
    print "... loading word embeddings"
    wvb = True if word_embeddings_fn.split('.')[-1] == 'bin' else False
    wv = gensim.models.KeyedVectors.load_word2vec_format(word_embeddings_fn, binary=wvb,
                                                         limit=150000)
    print "...... done"
    print "... calculating similarities"
    # If missing, give 1 to self and 0 else; give 0 similarity between in and out.
    pred_cosine = [[(1 + wv.similarity(predicates[pidx], predicates[pjdx])) / 2.0
                    if predicates[pjdx] in wv.vocab else 0
                    for pjdx in range(nb_predicates)]
                   if predicates[pidx] in wv.vocab else
                   [0 if pjdx != pidx else 1 for pjdx in range(nb_predicates)]
                   for pidx in range(nb_predicates)]
    print "...... done"
    print "... done"

    # Fit SVMs.
    print "fitting SVMs..."
    kappas = []  # pidx, b, m
    num_examples = []  # pidx
    for pidx in range(nb_predicates):
        if pidx not in valid_predicates:
            print "... '" + predicates[pidx] + "' insufficient labels"
            kappas.append({b: {m: 0 for _b, m in contexts if _b == b} for b, _ in contexts})
            num_examples.append(0)
            continue
        print "... '" + predicates[pidx] + "' fitting"
        train_pairs = [(oidx, labels[oidx][pidx]) for oidx in range(nb_objects)
                       if labels[oidx][pidx] == 0 or labels[oidx][pidx] == 1]
        num_examples.append(len(train_pairs))
        pc = {}
        pk = {}
        for b, m in contexts:
            if b not in pc:
                pc[b] = {}
                pk[b] = {}
            pc[b][m] = fit_classifier(kernel, b, m, train_pairs, object_feats)
            pk[b][m] = get_margin_kappa(pc[b][m], b, m, train_pairs, object_feats,
                                        xval=train_pairs, kernel=kernel)
        s = sum([pk[b][m] for b, m in contexts])
        for b, m in contexts:
            pk[b][m] = pk[b][m] / float(s) if s > 0 else 1.0 / nb_contexts
        kappas.append(pk)
    print "... done"

    print "writing labels.csv..."
    with open(os.path.join(outdir, 'labels.csv'), 'w') as f:
        h = ['oidx']
        h.extend(predicates)
        f.write(','.join(h) + '\n')
        for oidx in range(nb_objects):
            l = [str(oidx)]
            l.extend([str(int((labels[oidx][pidx] - 0.5) * 2)) for pidx in range(nb_predicates)])
            f.write(','.join(l) + '\n')
    print "... done"

    print "writing cosine.csv..."
    with open(os.path.join(outdir, 'cosine.csv'), 'w') as f:
        h = ['predicate']
        h.extend(predicates)
        f.write(','.join(h) + '\n')
        for pidx in range(nb_predicates):
            l = [predicates[pidx]]
            l.extend([str(pred_cosine[pidx][pjdx]) for pjdx in range(nb_predicates)])
            f.write(','.join(l) + '\n')
    print "... done"

    print "writing kappas.csv..."
    with open(os.path.join(outdir, 'kappas.csv'), 'w') as f:
        h = ['predicate']
        h.extend([';'.join(c) for c in contexts])
        f.write(','.join(h) + '\n')
        for pidx in range(nb_predicates):
            l = [predicates[pidx]]
            l.extend([str(kappas[pidx][c[0]][c[1]]) for c in contexts])
            f.write(','.join(l) + '\n')
    print "... done"

    print "writing behaviors.csv..."
    with open(os.path.join(outdir, 'behaviors.csv'), 'w') as f:
        h = ['predicate']
        h.extend(behaviors)
        f.write(','.join(h) + '\n')
        for pidx in range(nb_predicates):
            l = [predicates[pidx]]
            l.extend([str(behavior_annotations[pidx][b]) for b in behaviors])
            f.write(','.join(l) + '\n')
    print "... done"

    print "writing modalities.csv..."
    with open(os.path.join(outdir, 'modalities.csv'), 'w') as f:
        h = ['predicate']
        h.extend(modalities)
        f.write(','.join(h) + '\n')
        for pidx in range(nb_predicates):
            l = [predicates[pidx]]
            l.extend([str(modality_annotations[pidx][m] if pidx in modality_annotations else 0) for m in modalities])
            f.write(','.join(l) + '\n')
    print "... done"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True,
                        help="data directory")
    parser.add_argument('--kernel', type=str, required=True,
                        help="SVM kernel to use (linear, poly, rbf)")
    parser.add_argument('--word_embeddings', type=str, required=True,
                        help="word embeddings binary to use")
    parser.add_argument('--outdir', type=str, required=True,
                        help="directory to write out csvs")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
