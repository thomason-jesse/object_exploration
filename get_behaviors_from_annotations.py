#!/usr/bin/env python
__author__ = 'jesse'

import argparse
import operator
import os
import pickle
import numpy as np
import sys
from scipy.stats import entropy
from sklearn.metrics import cohen_kappa_score


def main():

    behaviors = ["drop", "grasp", "hold", "lift", "look", "lower", "press", "push"]

    # Convert flags to local variables.
    infile = FLAGS_infile
    indir = FLAGS_indir
    min_kappa = FLAGS_min_kappa
    outfile = FLAGS_outfile

    # Read in predicate pickle.
    with open(os.path.join(indir, "predicates.pickle"), 'rb') as f:
        predicates = pickle.load(f)

    # Read in annotations.
    subject_annotations = {}  # sid, pidx
    with open(infile, 'r') as f:
        for line in f.readlines()[1:]:
            sidstr, pred, predb = line.strip().split(';')
            sid = int(sidstr)
            pidx = predicates.index(pred)
            predbs = predb.split(',')
            for b in predbs:
                if b not in behaviors:
                    sys.exit("ERROR: unrecognized behavior '" + b + "'")
            pbs = [1 if b in predbs else 0 for b in behaviors]
            if sid not in subject_annotations:
                subject_annotations[sid] = {}
            assert pidx not in subject_annotations[sid]  # repeat sid, pidx
            subject_annotations[sid][pidx] = pbs

    # Calculate kappa values.
    annotator_scores_flat = []
    votes = [[[] for _ in range(len(behaviors))]
             for _ in range(len(predicates))]  # pidx, bidx
    for pidx in range(len(predicates)):
        for bidx in range(len(behaviors)):
            if len(annotator_scores_flat) == 0:
                annotator_scores_flat = [[] for _ in range(len(subject_annotations.keys()))]
            for aidx in range(len(annotator_scores_flat)):
                annotator_scores_flat[aidx].append(subject_annotations[aidx][pidx][bidx])
                votes[pidx][bidx].append(subject_annotations[aidx][pidx][bidx])
    ks = []
    annotator_ks = [[] for _ in range(len(annotator_scores_flat))]
    for aidx in range(len(annotator_scores_flat)):
        for ajdx in range(aidx+1, len(annotator_scores_flat)):
            k = cohen_kappa_score(annotator_scores_flat[aidx], annotator_scores_flat[ajdx])
            print "annotators " + str(aidx) + ", " + str(ajdx) + ": k=" + str(k)
            ks.append(k)
            annotator_ks[aidx].append(k)
            annotator_ks[ajdx].append(k)
    print "avg kappa: " + str(np.mean(ks))

    # Remove lower than threshold annotators.
    remove_annotators = []
    annotator_means = []
    for aidx in range(len(annotator_scores_flat)):
        m = np.mean(annotator_ks[aidx])
        annotator_means.append(m)
        if m < min_kappa:
            del subject_annotations[aidx]
            remove_annotators.append(aidx)
    print ("removed annotators " + str(remove_annotators) + " whose kappas " +
           str([annotator_means[aidx] for aidx in remove_annotators]) +
           " fell below threshold " + str(min_kappa))

    # Calculate kappa values.
    annotator_scores_flat = []
    annotators = subject_annotations.keys()
    votes = [[[] for _ in range(len(behaviors))]
             for _ in range(len(predicates))]  # pidx, bidx
    for pidx in range(len(predicates)):
        for bidx in range(len(behaviors)):
            if len(annotator_scores_flat) == 0:
                annotator_scores_flat = [[] for _ in range(len(subject_annotations.keys()))]
            for aidx in range(len(annotator_scores_flat)):
                annotator_scores_flat[aidx].append(subject_annotations[annotators[aidx]][pidx][bidx])
                votes[pidx][bidx].append(subject_annotations[annotators[aidx]][pidx][bidx])
    ks = []
    annotator_ks = [[] for _ in range(len(annotator_scores_flat))]
    for aidx in range(len(annotator_scores_flat)):
        for ajdx in range(aidx+1, len(annotator_scores_flat)):
            k = cohen_kappa_score(annotator_scores_flat[aidx], annotator_scores_flat[ajdx])
            print "annotators " + str(aidx) + ", " + str(ajdx) + ": k=" + str(k)
            ks.append(k)
            annotator_ks[aidx].append(k)
            annotator_ks[ajdx].append(k)
    print "avg kappa: " + str(np.mean(ks))

    # Print disagreements and decision directions.
    for pidx in range(len(predicates)):
        s = 0
        for bidx in range(len(behaviors)):
            m = np.mean(votes[pidx][bidx])
            votes[pidx][bidx] = m
            s += m
        votes[pidx] = [votes[pidx][bidx] / s for bidx in range(len(behaviors))]

    # Write annotation pickle outfile.
    with open(outfile, 'wb') as f:
        pickle.dump([{behaviors[bidx]: votes[pidx][bidx] for bidx in range(len(behaviors))}
                     for pidx in range(len(predicates))], f)

    # Calculate KL divergence between predicate behavior distributions.
    uniform = [1.0 / len(behaviors) for _ in behaviors]
    kls = {}
    for pidx in range(len(predicates)):
        kls[predicates[pidx]] = entropy([votes[pidx][bidx] for bidx in range(len(behaviors))], uniform)
    for key, value in sorted(kls.items(), key=operator.itemgetter(1)):
        print key, value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, required=True,
                        help="annotations csv")
    parser.add_argument('--indir', type=str, required=True,
                        help="data dir")
    parser.add_argument('--min_kappa', type=float, required=True,
                        help="minimum average kappa an annotator must have to remain in consideration")
    parser.add_argument('--outfile', type=str, required=True,
                        help="behavior annotations pickle")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
