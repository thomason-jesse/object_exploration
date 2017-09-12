#!/usr/bin/env python
__author__ = 'jesse'
''' borrowed from ispy_synsets/ and modified. Use to train SVM-based predicate classifiers on ispy or raw data.
'''

import argparse
import os
import pickle


def main():

    # Convert flags to local variables.
    results_dir = FLAGS_results_dir
    outfile = FLAGS_outfile
    behavior = FLAGS_behavior
    target_fn = "results.pickle" if behavior is None else "results.pickle." + behavior

    results_files = []
    for root, dirs, files in os.walk(results_dir):
        for fn in files:
            if fn == target_fn:
                results_files.append(os.path.join(root, fn))
    print results_files

    results_sums = {}
    for results_file in results_files:
        with open(results_file, 'rb') as f:
            r = pickle.load(f)
            for oidx in r:
                if oidx not in results_sums:
                    results_sums[oidx] = r[oidx]
                else:
                    results_sums[oidx] = [results_sums[oidx][pidx] + r[oidx][pidx] for pidx in range(len(r[oidx]))]

    results = {oidx: [results_sums[oidx][pidx] / float(len(results_files))
                      for pidx in range(len(results_sums[oidx]))]
               for oidx in results_sums}

    with open(outfile, 'wb') as f:
        pickle.dump(results, f)
    print "ensembled results for " + str(len(results_files)) + " models"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True,
                        help="directory with single-layer subs with results pickles")
    parser.add_argument('--outfile', type=str, required=True,
                        help="file to write ensembled results")
    parser.add_argument('--behavior', type=str, required=False,
                        help="get results for a particular behavior only when doing directory search")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
