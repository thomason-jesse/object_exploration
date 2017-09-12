#!/usr/bin/env python
__author__ = 'jesse'

import argparse
import os
import pickle
import numpy as np
import sys
from sklearn.metrics import cohen_kappa_score
from scipy.stats import mode


def main():

    # Convert flags to local variables.
    infile = FLAGS_infile
    indir = FLAGS_indir
    outfile = FLAGS_outfile

    # Read in predicate pickle.
    with open(os.path.join(indir, "predicates.pickle"), 'rb') as f:
        predicates = pickle.load(f)

    # Read in annotations.
    labels = {}
    annotators = []
    with open(infile, 'r') as f:
        lines = f.readlines()
        h = lines[0].strip().split(',')
        for l in lines[1:]:
            parts = l.strip().split(',')
            sidx = int(parts[h.index('sidx')])
            oidx = int(parts[h.index('oidx')])
            if oidx not in labels:
                labels[oidx] = {}
            if sidx not in annotators:
                annotators.append(sidx)
            unrecognized_preds = [p for p in parts[2:] if p not in predicates]
            if len(unrecognized_preds) > 0:
                sys.exit("ERROR: unrecognized predicates: " + str(unrecognized_preds))
            labels[oidx][sidx] = [1 if p in parts[2:] else 0 for p in predicates]

    # Calculate pairwise kappa values.
    kappas = []
    ks = 0
    for idx in range(len(annotators)):
        aidx = annotators[idx]
        for jdx in range(idx + 1, len(annotators)):
            ajdx = annotators[jdx]
            fi = []
            fj = []
            for oidx in labels:
                if aidx in labels[oidx].keys() and ajdx in labels[oidx].keys():
                    fi.extend(labels[oidx][aidx])
                    fj.extend(labels[oidx][ajdx])
            if len(fi) > 0:
                ka = cohen_kappa_score(fi, fj)
                kappas.append((aidx, ajdx, ka))
                print "annotators " + str(aidx) + ", " + str(ajdx) + ": k=" + str(ka)
                ks += ka
    print "avg kappa: " + str(ks / len(kappas))

    # Print disagreements and decision directions.
    v_labels = {}  # oidx, pidx, {0, 1}
    for oidx in labels:
        v_labels[oidx] = [0.5 for _ in range(len(predicates))]
        for pidx in range(len(predicates)):
            votes = [labels[oidx][sidx][pidx] for sidx in labels[oidx].keys()]
            m = mode(votes)[0][0]
            if m != np.mean(votes):
                print ("oidx " + str(oidx) + ", predicate " + predicates[pidx] +
                       " disagreement: " + str(votes) + " -> " + str(m))
            v_labels[oidx][pidx] = m

    # Write annotation pickle outfile.
    with open(outfile, 'wb') as f:
        pickle.dump(v_labels, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, required=True,
                        help="annotations csv")
    parser.add_argument('--indir', type=str, required=True,
                        help="data directory")
    parser.add_argument('--outfile', type=str, required=True,
                        help="labels pickle")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
