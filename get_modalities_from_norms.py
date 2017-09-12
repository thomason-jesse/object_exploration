#!/usr/bin/env python
__author__ = 'jesse'

import argparse
import os
import pickle
import numpy as np
import sys
from scipy.stats import mode
from sklearn.metrics import cohen_kappa_score


def main():

    lynott_modalities = ["AuditoryStrengthMean", "HapticStrengthMean", "VisualStrengthMean"]
    ispy_modalities = [["audio_ispy"], ["haptic_ispy"], ["color", "fpfh"]]

    # Convert flags to local variables.
    infile = FLAGS_infile
    indir = FLAGS_indir
    outfile = FLAGS_outfile

    # Read in predicate pickle.
    with open(os.path.join(indir, "predicates.pickle"), 'rb') as f:
        predicates = pickle.load(f)

    # Read in norms line by line and record relevant info.
    print "getting annotations..."
    modality_annotations = {}  # pidx, modality, distribution amount (sum 1 for pidx)
    with open(infile, 'r') as f:
        lines = f.readlines()
        h = lines[0].strip().split(',')
        for line in lines[1:]:
            p = line.split(',')
            w = p[h.index("Word")]
            if w in predicates:
                pidx = predicates.index(w)
                modality_annotations[pidx] = {}
                s = 0
                for idx in range(len(lynott_modalities)):
                    lm = float(p[h.index(lynott_modalities[idx])])
                    for im in ispy_modalities[idx]:
                        modality_annotations[pidx][im] = lm / len(ispy_modalities[idx])
                    s += lm
                for im in modality_annotations[pidx]:
                    modality_annotations[pidx][im] /= s

    print ("... done; found modalities for " + str(len(modality_annotations)) +
           " of " + str(len(predicates)) + " predicates")

    print "writing outfile..."
    with open(outfile, 'wb') as f:
        pickle.dump(modality_annotations, f)
    print "... done"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, required=True,
                        help="annotations csv")
    parser.add_argument('--indir', type=str, required=True,
                        help="data dir")
    parser.add_argument('--outfile', type=str, required=True,
                        help="behavior annotations pickle")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
