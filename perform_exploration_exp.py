#!/usr/bin/env python
__author__ = 'jesse'
''' Train mult-modal mahalanobis-based neighbor classifiers.
'''

import argparse
import gensim
import numpy as np
import os
import pickle
from functions import get_decisions_from_weights, get_data_for_classifier, fit_classifier, get_margin_kappa


def main():

    behaviors = ["drop", "grasp", "hold", "lift", "look", "lower", "press", "push"]
    modalities = ["audio_ispy", "color", "fpfh", "haptic_ispy"]  # "finger",
    nb_objects = 32

    # Convert flags to local variables.
    indir = FLAGS_indir
    test_oidx = FLAGS_test_oidx
    sparse_contexts = FLAGS_sparse_contexts
    kernel = FLAGS_kernel
    word_embeddings_fn = FLAGS_word_embeddings
    outdir = FLAGS_outdir
    retrain_svms = FLAGS_retrain_svms

    print "reading in folds, labels, predicates, and features..."
    # Read in folds.
    train_oidxs = [oidx for oidx in range(0, 32) if oidx != test_oidx]
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
                                for oidx in train_oidxs]) > 0
                        and sum([1 if labels[oidx][pidx] == 0 else 0
                                 for oidx in train_oidxs]) > 0]
    with open(os.path.join(indir, 'behavior_annotations.pickle'), 'rb') as f:
        behavior_annotations = pickle.load(f)
    print "... done"

    # Pre-calculate matrix of cosine similarity of word embeddings.
    print "pre-calculating word embeddings similarities..."
    print "... loading word embeddings"
    wv = gensim.models.KeyedVectors.load_word2vec_format(word_embeddings_fn, binary=True,
                                                         limit=150000)
    print "...... done"
    print "... calculating similarities"
    # If missing, give 1 to self and 0 else; give 0 similarity between in and out.
    pcc = 0.001  # pred_cosine values linearly up to prevent zero distributions; causes missing to back off to prior
    pred_cosine = [[(1 + wv.similarity(predicates[pidx], predicates[pjdx])) / 2.0 + pcc
                    if predicates[pjdx] in wv.vocab else pcc
                    for pjdx in range(nb_predicates)]
                   if predicates[pidx] in wv.vocab else
                   [pcc if pjdx != pidx else 1 for pjdx in range(nb_predicates)]
                   for pidx in range(nb_predicates)]

    # missing_preds = [predicates[pidx] for pidx in range(nb_predicates)
    #                  if predicates[pidx] not in wv.vocab]  # DEBUG
    # print missing_preds, len(missing_preds), len(predicates)  # DEBUG
    print "...... done"
    print "... done"

    # Fit SVMs.
    svm_fn = os.path.join(outdir, str(test_oidx) + '.svms.pickle')
    svms = kappas = num_examples = None
    try:
        with open(svm_fn, 'rb') as f:
            svms, kappas, num_examples = pickle.load(f)
        print "loaded SVMs from file"
    except IOError:
        retrain_svms = True
    if retrain_svms is not None:
        print "fitting SVMs for each predicate..."
        svms = []  # pidx, b, m
        kappas = []  # pidx, b, m
        num_examples = []  # pidx
        for pidx in range(nb_predicates):
            if pidx not in valid_predicates:
                print "... '" + predicates[pidx] + "' insufficient labels"
                svms.append(None)
                kappas.append({b: {m: 0 for _b, m in contexts if _b == b} for b, _ in contexts})
                num_examples.append(0)
                continue
            print "... '" + predicates[pidx] + "' fitting..."
            train_pairs = [(oidx, labels[oidx][pidx])
                           for oidx in train_oidxs
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
            svms.append(pc)
            s = sum([pk[b][m] for b, m in contexts])
            for b, m in contexts:
                pk[b][m] = pk[b][m] / float(s) if s > 0 else 1.0 / nb_contexts
            kappas.append(pk)
            print "...... done"
        print "... done"
        print "writing SVMs to file..."
        with open(svm_fn, 'wb') as f:
            pickle.dump([svms, kappas, num_examples], f)
        print "... done"

    # Get context SVM decisions for labeled predicates.
    print "getting decisions from context SVMs for labeled predicates..."
    test_pidxs = [pidx for pidx in range(nb_predicates)
                  if labels[test_oidx][pidx] == 0 or labels[test_oidx][pidx] == 1]
    test_context_ds = {}  # pidx, b, m
    for pidx in test_pidxs:
        test_context_ds[pidx] = {}
        for b, m in contexts:
            if b not in test_context_ds[pidx]:
                test_context_ds[pidx][b] = {}
            if svms[pidx] is not None:
                d = np.mean(svms[pidx][b][m].predict(get_data_for_classifier(b, m,
                                                                             [(test_oidx, labels[test_oidx][pidx])],
                                                                             object_feats)[0]))
            else:
                d = 0
            test_context_ds[pidx][b][m] = 1 if d > 0 else 0
    print "... done"

    # Use self kappas to mark object decisions.
    print "weighting decisions for self-estimated kappas for each predicate..."
    self_est_kappa_dec, self_est_kappa_bs = get_decisions_from_weights(test_context_ds,
                                                                       [[kappas[pidx][b][m] for b, m in contexts]
                                                                        for pidx in range(nb_predicates)],
                                                                       contexts)
    print "... done"

    # For varying numbers of examples, consider propagating kappa information to low resource predicates.
    print "running weighting schemes for increasing data requirements..."
    decision_matrix = {}  # required_n, scheme, pidx \in [0, 1]
    behavior_matrix = {}
    for required_n in range(2, 21):
        decision_matrix[required_n] = {}
        behavior_matrix[required_n] = {}
        required_mask = [0 if num_examples[pidx] < required_n else 1
                         for pidx in range(nb_predicates)]
        print ("...... " + str(sum(required_mask)) + " predicates available with "
               + str(required_n) + " training examples")

        # Calculate different sensorimotor context confidence schemes using leave-one-predicate-out xval.
        # Weight structure w is indexed first by name, then by pidx, then by cidx
        max_sims = [[i for i, x in enumerate(pred_cosine[pidx])
                     if x == np.max([pred_cosine[pidx][pjdx] * required_mask[pjdx]
                                     for pjdx in range(nb_predicates) if pjdx != pidx])]
                    for pidx in range(nb_predicates)]
        w = {"uniform": [[1.0 for _ in range(nb_contexts)] for _ in range(nb_predicates)],
             "prior_kappa": [[np.mean([kappas[pjdx][b][m] * required_mask[pjdx]
                                       for pjdx in range(nb_predicates) if pjdx != pidx])
                              for b, m in contexts] for pidx in range(nb_predicates)],
             "cos_avg_kappa": [[np.mean([kappas[pjdx][b][m] * pred_cosine[pidx][pjdx] * required_mask[pjdx]
                                         for pjdx in range(nb_predicates) if pjdx != pidx])
                                for b, m in contexts] for pidx in range(nb_predicates)],
             "cos_max_kappa": [[np.mean([kappas[pjdx][b][m] for pjdx in max_sims[pidx]])
                                for b, m in contexts] for pidx in range(nb_predicates)],
             "prior_ba": [[np.mean([behavior_annotations[pjdx][b] * required_mask[pjdx]
                                    for pjdx in range(nb_predicates) if pjdx != pidx])
                           for b, _ in contexts] for pidx in range(nb_predicates)],
             "cos_avg_ba": [[np.mean([behavior_annotations[pjdx][b] * pred_cosine[pidx][pjdx] * required_mask[pjdx]
                                      for pjdx in range(nb_predicates) if pjdx != pidx])
                             for b, m in contexts] for pidx in range(nb_predicates)],
             "cos_max_ba": [[np.mean([behavior_annotations[pjdx][b] for pjdx in max_sims[pidx]])
                             for b, m in contexts] for pidx in range(nb_predicates)]}
        for wn in w.keys():
            weight = w[wn]
            for pidx in range(nb_predicates):
                c = [weight[pidx][cidx] for cidx in range(nb_contexts)]
                s = sum(c)
                if sparse_contexts == 1 and s > 0:  # Takes maximum context only by changing distribution to it alone.
                    c = [c[idx] if c[idx] / float(s) >= 1.0 / float(len(c)) else 0
                         for idx in range(len(c))]
                    c = [c[idx] if c[idx] == max(c) else 0
                         for idx in range(len(c))]
                    s = sum(c)
                for cidx in range(nb_contexts):
                    if s > 0:
                        weight[pidx][cidx] = c[cidx] / float(s)
                    else:
                        weight[pidx][cidx] = 1.0 / float(len(c))
            w[wn] = weight

        # DEBUG - visualize distributional confidence scores
        # for pidx in range(nb_predicates):
        #     print predicates[pidx]
        #     d = {(b, m): w["cos_max_kappa"][pidx][contexts.index((b, m))] for b, m in contexts}
        #     for key, value in sorted(d.items(), key=operator.itemgetter(1)):
        #         print key, value
        #     _ = raw_input()
        # DEBUG

        # Use induced weights to mark object decisions.
        for wn in w.keys():
            decision_matrix[required_n][wn], behavior_matrix[required_n][wn] = \
                get_decisions_from_weights(test_context_ds, w[wn], contexts)

            # Enforce number of examples required, backing off to own kappas for predicates above limit.
            for pidx in decision_matrix[required_n][wn]:
                if required_mask[pidx] == 1:  # this predicate has enough examples to use own kappas
                    decision_matrix[required_n][wn][pidx] = self_est_kappa_dec[pidx]
                    behavior_matrix[required_n][wn][pidx] = self_est_kappa_bs[pidx]

    print "... done"

    # Write out decision matrix and self-estimated kappa decisions for each predicate.
    print "writing decisions to file..."
    with open(os.path.join(outdir, str(test_oidx) + '.pickle'), 'wb') as f:
        pickle.dump([decision_matrix, self_est_kappa_dec,
                     behavior_matrix, self_est_kappa_bs], f)
    print "... done"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True,
                        help="data directory")
    parser.add_argument('--test_oidx', type=int, required=True,
                        help="the object id held out for testing")
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
