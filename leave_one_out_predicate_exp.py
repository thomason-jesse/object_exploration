#!/usr/bin/env python
__author__ = 'jesse'

import argparse
import gensim
import numpy as np
import operator
import os
import pickle
import random
import sys
from functions import get_data_for_classifier, fit_classifier, get_margin_kappa_and_cm, get_margin_f1, get_margin_acc
from functions import get_best_sequence_of_behaviors, get_representatives_of_unique_sets, get_labels, get_kappa


def shuffle_ties(l):

    # Identify spans of equal values.
    spans = []
    curr_span = [0, 0]
    for idx in range(1, len(l)):
        if np.isclose(l[idx][1], l[curr_span[0]][1]):
            curr_span[1] = idx
        else:
            spans.append(curr_span)
            curr_span = [idx, idx]
    spans.append(curr_span)

    # Shuffle order of equal spans.
    for s in spans:
        c = l[s[0]:s[1] + 1]
        random.shuffle(c)
        l = l[:s[0]] + c + l[s[1] + 1:]

    return l


def main():

    behaviors = ["drop", "grasp", "hold", "lift", "look", "lower", "press", "push"]
    modalities = ["audio", "color", "fpfh", "haptics", "fc7"]
    nb_objects = 32
    nb_behaviors = len(behaviors)
    # time for setup + feature extraction
    behavior_t = {"drop": 9.8, "grasp": 22., "hold": 5.7, "lift": 11.1,
                  "look": 0.8, "lower": 10.6, "press": 22., "push": 22.}  # actual costs
    min_conf = 0
    nb_obs = 5  # an artifact of the dataset. there are 5 observations available per behavior

    # Convert flags to local variables.
    indir = FLAGS_indir
    test_pidx = FLAGS_test_pidx
    kernel = FLAGS_kernel
    word_embeddings_fn = FLAGS_word_embeddings
    required_examples = FLAGS_required_examples
    train_kappa_threshold = FLAGS_train_kappa_threshold
    outfile = FLAGS_outfile
    resample_test = FLAGS_resample_test
    alternative_labels = FLAGS_alternative_labels
    uniform_costs = FLAGS_uniform_costs
    degenerate = FLAGS_degenerate
    use_behavior_decisions = FLAGS_use_behavior_decisions

    if uniform_costs is not None and uniform_costs:
        behavior_t = {"drop": 1., "grasp": 1., "hold": 1., "lift": 1.,
                      "look": 1., "lower": 1., "press": 1., "push": 1.}  # uniform costs
    time_step = int(np.ceil(sum([behavior_t[b] for b in behaviors])))
    # time_step = int(np.ceil(sum([behavior_t[b] for b in behaviors]) / 5.))  # increase samples

    print "reading in folds, labels, predicates, and features..."
    labels = get_labels(indir, alternative_labels)

    # train/test split based on labels for predicate in question
    # f_train = range(nb_objects)
    # f_test = range(nb_objects)
    # fixed train/test splits from fully annotated sets plus leftovers
    f_train = [10, 3, 27, 7] + [18, 2, 20, 17] + [5, 14, 8, 15] + [1, 30, 29, 31]
    f_test = [21, 24, 19, 23] + [16, 0, 4, 9] + [22, 28, 12, 25] + [11, 6, 26, 13]

    # DEBUG - imbalanced train 24 / test 8
    # f_train = [10, 3, 27, 7] + [18, 2, 20, 17] + [5, 14, 8, 15] + [1, 30, 29, 31] + [21, 24, 19, 23] + [16, 0, 4, 9]
    # f_test = [22, 28, 12, 25] + [11, 6, 26, 13]

    test_oidxs = [oidx for oidx in f_test if labels[oidx][test_pidx] == 0 or labels[oidx][test_pidx] == 1]
    train_oidxs = [oidx for oidx in f_train if oidx not in test_oidxs]

    with open(os.path.join(indir, 'predicates.pickle'), 'rb') as f:
        predicates = pickle.load(f)
        nb_predicates = len(predicates)
    feature_fn = os.path.join(indir, 'features.pickle')
    with open(feature_fn, 'rb') as f:
        object_feats = pickle.load(f)
    contexts = []
    for oidx in range(nb_objects):
        contexts = []
        for b in behaviors:
            if b not in object_feats[oidx]:
                continue
            for m in modalities:
                if m not in object_feats[oidx][b]:
                    continue
                contexts.append((b, m))
    nb_contexts = len(contexts)
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
    if len(trainable_preds) == 0:
        print "WARNING: no trainable predicates under constraints"
        sys.exit()
    with open(os.path.join(indir, 'behavior_annotations.pickle'), 'rb') as f:
        behavior_annotations = pickle.load(f)
    print "... done"

    # DEBUG
    # for oidx in range(nb_objects):  # DEBUG
    #     print str(oidx) + ": " + ', '.join([predicates[pidx] for pidx in range(nb_predicates)
    #                                        if labels[oidx][pidx] == 1])  # DEBUG
    # _ = raw_input()  # DEBUG
    # DEBUG

    # Pre-calculate matrix of cosine similarity of word embeddings.
    print "pre-calculating word embeddings similarities..."
    print "... loading word embeddings"
    wvb = True if word_embeddings_fn.split('.')[-1] == 'bin' else False
    wv = gensim.models.KeyedVectors.load_word2vec_format(word_embeddings_fn, binary=wvb,
                                                         limit=150000)
    print "...... done"
    print "... calculating similarities"
    # If missing, give 1 to self and 0 else; give 0 similarity between in and out.
    if predicates[test_pidx] in wv.vocab:
        pred_cosine = [(1 + wv.similarity(predicates[test_pidx], predicates[pjdx])) / 2.0
                       if predicates[pjdx] in wv.vocab else 0 for pjdx in range(nb_predicates)]
    else:
        pred_cosine = [0 if pjdx != test_pidx else 1 for pjdx in range(nb_predicates)]
    pred_cosine = [val / sum(pred_cosine) for val in pred_cosine]
    max_sims = [i for i, x in enumerate(pred_cosine)
                if np.isclose(x, np.max([pred_cosine[pjdx] for pjdx in trainable_preds]))]
    print "max sims '" + str(predicates[test_pidx]) + "': " + ','.join([predicates[midx] for midx in max_sims])
    top_k_sims = max_sims[:]
    while len(top_k_sims) < 3:  # get top 3
        curr_max_val = np.max([pred_cosine[pjdx] for pjdx in trainable_preds
                               if pjdx not in top_k_sims])
        top_k_sims.extend([i for i, x in enumerate(pred_cosine)
                           if np.isclose(x, curr_max_val)])
    print "top k sims '" + str(predicates[test_pidx]) + "': " + ','.join([predicates[midx] for midx in top_k_sims])

    # print len(wv.vocab)  # DEBUG
    # print [predicates[pidx] for pidx in range(nb_predicates) if predicates[pidx] in wv.vocab]  # DEBUG
    # missing_preds = [predicates[pidx] for pidx in range(nb_predicates)
    #                  if predicates[pidx] not in wv.vocab]  # DEBUG
    # print missing_preds, len(missing_preds), len(predicates)  # DEBUG
    print "...... done"
    print "... done"

    # Fit SVMs.
    print "fitting SVMs for each train predicate considering only objects not labeled for test predicate..."
    kappas = []  # pidx, b, m
    behavior_kappas = []
    f1s = []
    accs = []
    num_examples = []  # pidx
    for pidx in range(nb_predicates):
        if pidx not in trainable_preds:
            print "... '" + predicates[pidx] + "' insufficient labels or is test pred"
            kappas.append({b: {m: 0 for _b, m in contexts if _b == b} for b, _ in contexts})
            behavior_kappas.append({b: 0 for b in behaviors})
            f1s.append({b: {m: 0 for _b, m in contexts if _b == b} for b, _ in contexts})
            accs.append({b: {m: 0 for _b, m in contexts if _b == b} for b, _ in contexts})
            num_examples.append(0)
            continue
        print "... '" + predicates[pidx] + "' fitting"
        train_pairs = [(oidx, labels[oidx][pidx])
                       for oidx in train_oidxs
                       if labels[oidx][pidx] == 0 or labels[oidx][pidx] == 1]
        num_examples.append(len(train_pairs))
        pc = {}
        pcm = {}
        pk = {}
        pf = {}
        pa = {}
        for b, m in contexts:
            if b not in pc:
                pc[b] = {}
                pcm[b] = {}
                pk[b] = {}
                pf[b] = {}
                pa[b] = {}
            pc[b][m] = fit_classifier(kernel, b, m, train_pairs, object_feats)
            pk[b][m], pcm[b][m] = get_margin_kappa_and_cm(pc[b][m], b, m, train_pairs, object_feats,
                                                          xval=train_pairs, kernel=kernel, minc=(min_conf, 0))
            pf1 = get_margin_f1(pc[b][m], b, m, train_pairs, object_feats,
                                xval=train_pairs, kernel=kernel, minc=(min_conf, 0))
            # pf[b][m] = pf1 + 1. / nb_contexts  # test; adds uniform conf vector to all others effectively
            pf[b][m] = pf1
            pa[b][m] = get_margin_acc(pc[b][m], b, m, train_pairs, object_feats,
                                      xval=train_pairs, kernel=kernel, minc=(min_conf, 0))
        # print "pidx: " + str(pidx)  # DEBUG
        # print '\n'.join(['\t'.join([b, m, str(pk[b][m]), str(pf[b][m])]) for b, m in contexts])  # DEBUG
        # _ = raw_input()  # DEBUG
        kappas.append(pk)
        pbk = {}
        for b in behaviors:
            bcm = [[sum([pk[b][m] * pcm[b][m][0][0] for m in pk[b].keys()]),
                    sum([pk[b][m] * pcm[b][m][0][1] for m in pk[b].keys()])],
                   [sum([pk[b][m] * pcm[b][m][1][0] for m in pk[b].keys()]),
                    sum([pk[b][m] * pcm[b][m][1][1] for m in pk[b].keys()])]]
            pbk[b] = get_kappa(bcm)
        behavior_kappas.append(pbk)
        f1s.append(pf)
        accs.append(pa)
    print "... done"

    # Propagate kappas to held out predicate under various schemes.
    print "calculating confidence distributions..."
    w = {"uniform": [1.0 for _ in range(nb_contexts)],
         "prior_kappa": [np.mean([kappas[pjdx][b][m]
                                 for pjdx in range(nb_predicates) if pjdx != test_pidx])
                         for b, m in contexts],
         "cos_avg_kappa": [np.mean([kappas[pjdx][b][m] * pred_cosine[pjdx]
                                    for pjdx in range(nb_predicates) if pjdx != test_pidx])
                           for b, m in contexts],
         "cos_max_kappa": [np.mean([kappas[pjdx][b][m] for pjdx in max_sims])
                           for b, m in contexts],
         "cos_top3_kappa": [np.mean([kappas[pjdx][b][m] * pred_cosine[pjdx]
                                     for pjdx in top_k_sims])
                           for b, m in contexts],
         # "prior_f1": [np.mean([f1s[pjdx][b][m]
         #                       for pjdx in range(nb_predicates) if pjdx != test_pidx])
         #              for b, m in contexts],
         # "cos_avg_f1": [np.mean([f1s[pjdx][b][m] * pred_cosine[pjdx]
         #                         for pjdx in range(nb_predicates) if pjdx != test_pidx])
         #                for b, m in contexts],
         # "cos_max_f1": [np.mean([f1s[pjdx][b][m] for pjdx in max_sims])
         #                for b, m in contexts],
         # "prior_acc": [np.mean([accs[pjdx][b][m]
         #                       for pjdx in range(nb_predicates) if pjdx != test_pidx])
         #               for b, m in contexts],
         # "cos_max_acc": [np.mean([accs[pjdx][b][m] for pjdx in max_sims])
         #                 for b, m in contexts],
         # "prior_ba": [np.mean([behavior_annotations[pjdx][b]
         #                       for pjdx in range(nb_predicates) if pjdx != test_pidx])
         #              for b, _ in contexts],
         # "cos_avg_ba": [np.mean([behavior_annotations[pjdx][b] * pred_cosine[pjdx]
         #                         for pjdx in range(nb_predicates) if pjdx != test_pidx])
         #                for b, _ in contexts],
         # "cos_max_ba": [np.mean([behavior_annotations[pjdx][b] for pjdx in max_sims])
         #                for b, _ in contexts],
         # "cos_max_kba": [np.mean([behavior_annotations[pjdx][b] * kappas[pjdx][b][m] for pjdx in max_sims])
         #                 for b, m in contexts],
         # "cos_max_fba": [np.mean([behavior_annotations[pjdx][b] * f1s[pjdx][b][m] for pjdx in max_sims])
         #                 for b, m in contexts],
         # "cos_max_aba": [np.mean([behavior_annotations[pjdx][b] * accs[pjdx][b][m] for pjdx in max_sims])
         #                 for b, m in contexts],
         "ba": [behavior_annotations[test_pidx][b] for b, _ in contexts],
         "kba": [np.mean([kappas[pjdx][b][m] * pred_cosine[pjdx]
                          for pjdx in top_k_sims]) * behavior_annotations[test_pidx][b]
                 for b, m in contexts]
         }
    wb = {"uniform": [1.0 for _ in range(nb_behaviors)],
          "prior_kappa": [np.mean([behavior_kappas[pjdx][b]
                                  for pjdx in range(nb_predicates) if pjdx != test_pidx])
                          for b in behaviors],
          "cos_avg_kappa": [np.mean([behavior_kappas[pjdx][b] * pred_cosine[pjdx]
                                     for pjdx in range(nb_predicates) if pjdx != test_pidx])
                            for b in behaviors],
          "cos_max_kappa": [np.mean([behavior_kappas[pjdx][b] for pjdx in max_sims])
                            for b in behaviors],
          "cos_top3_kappa": [np.mean([behavior_kappas[pjdx][b] * pred_cosine[pjdx]
                                      for pjdx in top_k_sims])
                             for b in behaviors],
          # "prior_ba": [np.mean([behavior_annotations[pjdx][b]
          #                      for pjdx in range(nb_predicates) if pjdx != test_pidx])
          #             for b in behaviors],
          # "cos_max_ba": [np.mean([behavior_annotations[pjdx][b] for pjdx in max_sims])
          #                for b in behaviors],
          # "cos_max_kba": [np.mean([behavior_annotations[pjdx][b] * behavior_kappas[pjdx][b] for pjdx in max_sims])
          #                 for b in behaviors]
          "ba": [behavior_annotations[test_pidx][b] for b in behaviors],
          "kba": [np.mean([behavior_kappas[pjdx][b] * pred_cosine[pjdx]
                           for pjdx in top_k_sims]) * behavior_annotations[test_pidx][b]
                  for b in behaviors]
          }
    # w = {"uniform": w["uniform"]}  # DEBUG
    # wb = {"uniform": wb["uniform"]}  # DEBUG
    print "... done"

    # DEBUG - visualize distributional confidence scores
    # d = {(b, m): w["ba"][contexts.index((b, m))] for b, m in contexts}
    # for key, value in sorted(d.items(), key=operator.itemgetter(1)):
    #     print key, value
    # _ = raw_input()
    # DEBUG

    # Perform leave-one-object-out training/testing using exploration policies elicited by weight distributions.
    # Record decision on held-out object and behaviors used to arrive at it during training/testing.
    print "performing leave-one-object-out cross validation with training/testing guided by kappa weights..."
    max_allowed = int(np.ceil(sum([behavior_t[b] * nb_obs for b in behaviors]))) + 1
    min_allowed = int(np.ceil(min([behavior_t[b] * nb_obs for b in behaviors])))
    # min_allowed = max_allowed - 1  # DEBUG
    train_contexts_used = {wn: {dependent_inc: {test_oidx: [] for test_oidx in test_oidxs}
                                for dependent_inc in range(min_allowed, max_allowed, time_step) + [max_allowed - 1]}
                           for wn in w.keys()}
    train_behaviors_used = {wn: {dependent_inc: {test_oidx: [] for test_oidx in test_oidxs}
                                 for dependent_inc in range(min_allowed, max_allowed, time_step) + [max_allowed - 1]}
                            for wn in w.keys()}
    train_time_used = {wn: {dependent_inc: {test_oidx: [] for test_oidx in test_oidxs}
                            for dependent_inc in range(min_allowed, max_allowed, time_step) + [max_allowed - 1]}
                       for wn in w.keys()}
    test_contexts_used = {wn: {dependent_inc: {test_oidx: [] for test_oidx in test_oidxs}
                               for dependent_inc in range(min_allowed, max_allowed, time_step) + [max_allowed - 1]}
                          for wn in w.keys()}
    test_behaviors_used = {wn: {dependent_inc: {test_oidx: [] for test_oidx in test_oidxs}
                                for dependent_inc in range(min_allowed, max_allowed, time_step) + [max_allowed - 1]}
                           for wn in w.keys()}
    test_time_used = {wn: {dependent_inc: {test_oidx: [] for test_oidx in test_oidxs}
                           for dependent_inc in range(min_allowed, max_allowed, time_step) + [max_allowed - 1]}
                      for wn in w.keys()}
    decisions = {wn: {dependent_inc: {test_oidx: [] for test_oidx in test_oidxs}
                      for dependent_inc in range(min_allowed, max_allowed, time_step) + [max_allowed - 1]}
                 for wn in w.keys()}
    for test_oidx in test_oidxs:
        print "... leaving " + str(test_oidx) + " out of " + str(test_oidxs)

        # For varying thresholds, record number of behaviors, contexts, and test result on held out object.
        for wn in w.keys():

            if use_behavior_decisions:
                behavior_r = {behaviors[bidx]: wb[wn][bidx] for bidx in range(nb_behaviors)}
            else:
                behavior_r = {behaviors[bidx]: sum([w[wn][cidx]
                                                    for cidx in range(nb_contexts)
                                                    if contexts[cidx][0] == behaviors[bidx]])
                              for bidx in range(nb_behaviors)}
            s = sum([behavior_r[b] for b in behavior_r])
            behavior_r = {b: behavior_r[b] / s if s > 0 else 1.0 / len(behavior_r.keys())
                          for b in behavior_r}

            # print wn, behavior_r  # DEBUG
            # _ = raw_input()  # DEBUG

            for dependent_inc in range(min_allowed, max_allowed, time_step) + [max_allowed - 1]:

                # Sample a training sequence, then a testing sequence based on that
                for samples in range(100):
                    b_sequences = get_best_sequence_of_behaviors(behaviors, behavior_r, behavior_t, dependent_inc,
                                                                 maximize_reward=False, max_samples=100,
                                                                 random_walk=True, degenerate=degenerate)
                    b_sequences = get_representatives_of_unique_sets(b_sequences)
                    # print wn, dependent_inc, len(b_sequences)  # DEBUG
                    # print "\t\n".join([str(bseq) for bseq in b_sequences])  # DEBUG
                    # _ = raw_input()  # DEBUG

                    bsidx = random.randint(0, len(b_sequences) - 1)
                    b_sequence = b_sequences[bsidx]

                    # DEBUG - visualize distributional confidence scores
                    # print "training sequence and behavior reward distribution:"
                    # print b_sequence
                    # d = {b: behavior_r[b] for b in behaviors}
                    # for key, value in sorted(d.items(), key=operator.itemgetter(1)):
                    #     print key, value
                    # _ = raw_input()
                    # DEBUG

                    # Train SVMs on test_pidx objects except held-out test_oidx,
                    # getting local confidence estimates based on the behavior sequence in question.
                    # This allows us to train on a # of observations corresponding to the number of times
                    # each behavior was actually performed during training.
                    xval_svms = []  # cidx
                    xval_kappas = []
                    xval_cms = []
                    train_pairs = [(oidx, labels[oidx][test_pidx])
                                   for oidx in test_oidxs if oidx != test_oidx]
                    mc = (1 if len([l for _, l in train_pairs if l == 1]) >
                          len([l for _, l in train_pairs if l == 0]) else 0)
                    for cidx in range(nb_contexts):
                        b, m = contexts[cidx]
                        ol = b_sequence.count(b)
                        if ol > 0:
                            xval_svms.append(fit_classifier(kernel, b, m, train_pairs, object_feats,
                                                            obs_limit=ol))
                            xvk, xvcm = get_margin_kappa_and_cm(xval_svms[-1], b, m, train_pairs, object_feats,
                                                                xval=train_pairs, kernel=kernel,
                                                                minc=(min_conf, 0), obs_limit=ol)
                            xval_kappas.append(xvk)
                            xval_cms.append(xvcm)
                        else:
                            xval_svms.append(None)
                            xval_kappas.append(0)
                            xval_cms.append([[0, 0], [0, 0]])
                    xval_behavior_kappas = []
                    for b in behaviors:
                        if b in b_sequence:
                            bcm = [[sum([xval_kappas[cidx] * xval_cms[cidx][0][0] for cidx in range(nb_contexts)
                                         if contexts[cidx][0] == b]),
                                    sum([xval_kappas[cidx] * xval_cms[cidx][0][1] for cidx in range(nb_contexts)
                                         if contexts[cidx][0] == b])],
                                   [sum([xval_kappas[cidx] * xval_cms[cidx][1][0] for cidx in range(nb_contexts)
                                         if contexts[cidx][0] == b]),
                                    sum([xval_kappas[cidx] * xval_cms[cidx][1][1] for cidx in range(nb_contexts)
                                         if contexts[cidx][0] == b])]]
                            if bcm[0][0] + bcm[0][1] + bcm[1][0] + bcm[1][1] > 0:
                                xval_behavior_kappas.append(get_kappa(bcm))
                            else:
                                xval_behavior_kappas.append(0)
                        else:
                            xval_behavior_kappas.append(0)

                    # print dependent_inc, b_sequence, xval_behavior_kappas  # DEBUG
                    # _ = raw_input()  # DEBUG

                    # Train.
                    train_contexts = []
                    train_behaviors = []
                    for b in b_sequence:
                        bidx = behaviors.index(b)
                        train_contexts.extend([cidx for cidx in range(nb_contexts)
                                               if contexts[cidx][0] == behaviors[bidx]])
                        train_behaviors.append(bidx)
                    if (True and dependent_inc == max_allowed - 1 and
                            set(train_behaviors) != set(range(len(behaviors)))):  # DEBUG
                        print "WARNING: training set not maxed despite having enough time"  # DEBUG
                        print wn, dependent_inc, [behaviors[bidx] for bidx in train_behaviors]  # DEBUG
                        print "behavior_r: " + str(behavior_r)  # DEBUG
                        # _ = raw_input()  # DEBUG
                    train_contexts_used[wn][dependent_inc][test_oidx].append(train_contexts)
                    train_behaviors_used[wn][dependent_inc][test_oidx].append(train_behaviors)
                    train_time_used[wn][dependent_inc][test_oidx].append(sum([behavior_t[behaviors[bidx]]
                                                                              for bidx in train_behaviors]))

                    # Test.
                    contexts_for_sequence = [cjdx for cjdx in range(nb_contexts)
                                             if contexts[cjdx][0] in b_sequence]
                    s = sum([w[wn][cjdx] for cjdx in contexts_for_sequence])
                    context_weights = {cjdx: w[wn][cjdx] / s if s > 0 else 1.0 / len(contexts_for_sequence)
                                       for cjdx in contexts_for_sequence}

                    # At test time, re-order in-behavior contexts with own kappas.
                    # Allows re-ordering train behaviors too.
                    if resample_test:
                        # TODO: get this up-to-date with the use_behavior_decisions option, which is not
                        # TODO: currently implemented here
                        new_weights = []
                        dws = []
                        new_b_weights = {b: 0 for b in b_sequence}
                        for cidx in context_weights.keys():
                            xval_context_weight = xval_kappas[cidx]
                            # xval_context_weight = xval_f1s[cidx]
                            new_weights.append((cidx, xval_context_weight))
                            dws.append(xval_context_weight)
                        new_weights = {cidx: dw / sum(dws) if sum(dws) > 0 else 1.0 / len(new_weights)
                                       for cidx, dw in new_weights}
                        for cidx in new_weights:
                            new_b_weights[contexts[cidx][0]] += new_weights[cidx]
                        new_t = {b: behavior_t[b] for b in b_sequence}

                        # Get new sequences, this time maximizing for reward, e.g. taking high-reward actions first,
                        # with sufficiently high reward allowing taking vacuous actions between/after
                        # Additionally, allow time * nb_obs, that is, the policy expands to allow intermingling of
                        # different behaviors if it's greedy-optimal to do so, as well as repeating high-reward
                        # loops like, for example, 'grasp'/'drop'
                        new_sequences = get_best_sequence_of_behaviors(list(set(b_sequence)), new_b_weights,
                                                                       new_t, dependent_inc,
                                                                       maximize_reward=True,
                                                                       max_samples=100,
                                                                       random_walk=True,
                                                                       degenerate=degenerate)
                        if False and (len(new_sequences) > 1 or new_sequences[0] != b_sequence):  # DEBUG
                            print "context_weights: " + str(context_weights)  # DEBUG
                            print "behavior_r: " + str(behavior_r)  # DEBUG
                            print "b_sequence: " + str(b_sequence)  # DEBUG
                            print "new_weights: " + str([[contexts[cidx], new_weights[cidx]]
                                                         for cidx in new_weights.keys()])
                            print "new_b_weights: " + str(new_b_weights)  # DEBUG
                            print "new_sequences: " + str(new_sequences)  # DEBUG
                            _ = raw_input()  # DEBUG
                        context_weights = new_weights
                        test_sequences = new_sequences

                        # Get observations for each context and keep them in storage.
                        observations = []  # indexed by cidx; holds vectors of observations for each context
                        next_obs = [0 for _ in range(nb_behaviors)]  # holds index of next observation to retrieve
                        for cidx in range(nb_contexts):
                            observations.append(get_data_for_classifier(contexts[cidx][0], contexts[cidx][1],
                                                [(test_oidx, labels[test_oidx][test_pidx])], object_feats)[0])

                        tidx = random.randint(0, len(test_sequences) - 1)

                        # DEBUG - visualize distributional confidence scores
                        # print "testing sequence and behavior reward distribution:"
                        # print test_sequences[tidx]
                        # d = {b: new_b_weights[b] for b in list(set(b_sequence))}
                        # for key, value in sorted(d.items(), key=operator.itemgetter(1)):
                        #     print key, value
                        # _ = raw_input()
                        # DEBUG

                        for bidx in range(nb_behaviors):
                            next_obs[bidx] = 0
                        behaviors_so_far = []
                        contexts_so_far = []
                        pos = 0
                        neg = 0
                        for btidx in range(len(test_sequences[tidx])):
                            if sum([new_b_weights[b] if next_obs[behaviors.index(b)] < nb_obs else 0
                                    for b in test_sequences[tidx][:btidx + 1]]) == 0:
                                print "WARNING: aborting sequence early" + str(test_sequences[tidx])
                                break
                            b = test_sequences[tidx][btidx]
                            bidx = behaviors.index(b)
                            behaviors_so_far.append(bidx)
                            contexts_for_behavior = [cjdx for cjdx in range(nb_contexts)
                                                     if contexts[cjdx][0] == behaviors[bidx]]
                            context_weight_pairs = sorted({contexts_for_behavior[idx]:
                                                           context_weights[contexts_for_behavior[idx]]
                                                           for idx in range(len(contexts_for_behavior))}.items(),
                                                          key=operator.itemgetter(1), reverse=True)
                            context_weight_pairs = shuffle_ties(context_weight_pairs)
                            sufficient_dec_weight = False
                            for cidx, cdw in context_weight_pairs:
                                if cdw > 0:
                                    contexts_so_far.append(cidx)
                                    ds = xval_svms[cidx].predict([observations[cidx][next_obs[bidx]]])
                                    for d in ds:
                                        if d > 0:
                                            pos += d * cdw / test_sequences[tidx].count(b)
                                        else:
                                            neg -= d * cdw / test_sequences[tidx].count(b)
                                    # print pos, neg  # DEBUG
                                    # print test_oidx, contexts[cidx], ds, cdw  # DEBUG
                                    if pos > 0.5 or neg > 0.5:
                                        sufficient_dec_weight = True
                                        break
                            next_obs[bidx] += 1
                            if sufficient_dec_weight:
                                break
                        dec = 1 if pos > neg and (pos > 0 or neg > 0) else mc
                        test_contexts_used[wn][dependent_inc][test_oidx].append(contexts_so_far)
                        test_behaviors_used[wn][dependent_inc][test_oidx].append(behaviors_so_far)
                        test_time_used[wn][dependent_inc][test_oidx].append(sum([behavior_t[behaviors[bidx]]
                                                                                 for bidx in behaviors_so_far]))
                        decisions[wn][dependent_inc][test_oidx].append(dec)

                    else:

                        # Get observations for each context and keep them in storage.
                        observations = []  # indexed by cidx; holds vectors of observations for each context
                        next_obs = [0 for _ in range(nb_behaviors)]  # holds index of next observation to retrieve
                        for cidx in range(nb_contexts):
                            observations.append(get_data_for_classifier(contexts[cidx][0], contexts[cidx][1],
                                                [(test_oidx, labels[test_oidx][test_pidx])], object_feats)[0])

                        behaviors_so_far = []
                        contexts_so_far = []
                        dsum = 0
                        for btidx in range(len(b_sequence)):
                            b = b_sequence[btidx]
                            bidx = behaviors.index(b)
                            behaviors_so_far.append(bidx)
                            contexts_for_behavior = [cjdx for cjdx in range(nb_contexts)
                                                     if contexts[cjdx][0] == behaviors[bidx]]
                            if use_behavior_decisions:
                                bd = 0
                                for cidx in contexts_for_behavior:
                                    contexts_so_far.append(cidx)
                                    ds = xval_svms[cidx].predict([observations[cidx][next_obs[bidx]]])
                                    for d in ds:
                                        bd += d * xval_kappas[cidx]
                                bd = 1 if bd > 0 else -1
                                dsum += bd * xval_behavior_kappas[bidx] / b_sequence.count(b)
                            else:
                                for cidx in contexts_for_behavior:
                                    contexts_so_far.append(cidx)
                                    ds = xval_svms[cidx].predict([observations[cidx][next_obs[bidx]]])
                                    for d in ds:
                                        dsum += d * xval_kappas[cidx] / b_sequence.count(b)
                                        # print contexts[cidx], d, xval_kappas[cidx], b_sequence.count(b)  # DEBUG
                            next_obs[bidx] += 1
                        dec = 1 if dsum > 0 else 0
                        # print dec, dsum  # DEBUG
                        # _ = raw_input()  # DEBUG
                        test_contexts_used[wn][dependent_inc][test_oidx].append(contexts_so_far)
                        test_behaviors_used[wn][dependent_inc][test_oidx].append(behaviors_so_far)
                        test_time_used[wn][dependent_inc][test_oidx].append(sum([behavior_t[behaviors[bidx]]
                                                                                 for bidx in behaviors_so_far]))
                        decisions[wn][dependent_inc][test_oidx].append(dec)

                    # print test_oidx, dependent_inc, bsidx, tidx  # DEBUG
                    # print "contexts: " + str(test_contexts_used[wn][dependent_inc][test_oidx][bsidx])  # DEBUG
                    # print "behaviors: " + str(test_behaviors_used[wn][dependent_inc][test_oidx][bsidx])  # DEBUG
                    # print "times: " + str(test_time_used[wn][dependent_inc][test_oidx][bsidx])  # DEBUG
                    # print "decisions: " + str(decisions[wn][dependent_inc][test_oidx][bsidx])  # DEBUG
                    # _ = raw_input()  # DEBUG
    print "... done"

    # Write outfiles.
    print "writing outfile..."
    with open(outfile, 'wb') as f:
        pickle.dump([w.keys(), train_contexts_used, train_behaviors_used, train_time_used,
                     test_contexts_used, test_behaviors_used, test_time_used, decisions], f)
    print "... done"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True,
                        help="data directory")
    parser.add_argument('--test_pidx', type=int, required=True,
                        help="the predicate id held out for testing")
    parser.add_argument('--kernel', type=str, required=True,
                        help="SVM kernel to use (linear, poly, rbf)")
    parser.add_argument('--word_embeddings', type=str, required=True,
                        help="word embeddings binary to use")
    parser.add_argument('--required_examples', type=int, required=True,
                        help="how many positive and negative examples per predicate to qualify")
    parser.add_argument('--train_kappa_threshold', type=int, required=True,
                        help="number of examples required before trusting a training predicate classifier")
    parser.add_argument('--outfile', type=str, required=True,
                        help="file to write results pickle")
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
