#!/usr/bin/env python
__author__ = 'jesse'
''' borrowed from ispy_synsets/ and modified. Use to train SVM-based predicate classifiers on ispy or raw data.
'''

import argparse
import numpy as np
import operator
import os
import pickle
from sklearn.metrics import mean_squared_error
from scipy.stats import ttest_ind


def get_p_r_f1(cm):

    p = float(cm[1][1]) / (cm[1][1] + cm[0][1]) if (cm[1][1] + cm[0][1]) > 0 else 0
    r = float(cm[1][1]) / (cm[1][1] + cm[1][0]) if (cm[1][1] + cm[1][0]) > 0 else 0
    f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
    return p, r, f1


def get_kappa(cm):

    s = float(cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
    po = (cm[1][1] + cm[0][0]) / s
    ma = (cm[1][1] + cm[1][0]) / s
    mb = (cm[0][0] + cm[0][1]) / s
    pe = (ma + mb) / s
    kappa = (po - pe) / (1 - pe)
    return max(0, kappa)


def main():

    # Convert flags to local variables.
    indir = FLAGS_indir
    test_fold = FLAGS_test_fold
    results_file_dirs = FLAGS_results_file.split(',')
    svm_examples = FLAGS_svm_examples
    labels_fn = (FLAGS_alternative_labels if FLAGS_alternative_labels is not None
                 else os.path.join(indir, "labels.pickle"))
    behavior = FLAGS_behavior
    verbose = FLAGS_verbose if FLAGS_verbose is not None else 0
    target_fn = "results.pickle" if behavior is None else "results.pickle." + behavior

    # Read in data.
    print "reading in test oidxs and predicates..."
    with open(os.path.join(indir, "folds.pickle"), 'rb') as f:
        folds = pickle.load(f)
    with open(os.path.join(indir, "predicates.pickle"), 'rb') as f:
        predicates = pickle.load(f)
    with open(os.path.join(indir, "labels.pickle"), 'rb') as f:
        labels = pickle.load(f)

    # Blacklist predicates that aren't svm valid, if asked.
    if test_fold != -1:
        test_oidxs = folds[test_fold]
        predicates_to_evaluate = []
        for pidx in range(0, len(predicates)):
            num_pos = sum([1 for oidx in range(0, 32) if oidx not in test_oidxs and labels[oidx][pidx] == 1])
            num_neg = sum([1 for oidx in range(0, 32) if oidx not in test_oidxs and labels[oidx][pidx] == 0])
            if num_pos >= svm_examples and num_neg >= svm_examples:
                predicates_to_evaluate.append(pidx)
        print "num objects:\t" + str(len(test_oidxs))
        print "num preds:\t" + str(len(predicates_to_evaluate))

    # Load alternative labels for evaluation.
    with open(labels_fn, 'rb') as f:
        labels = pickle.load(f)
    print "... done"

    rfr = {}
    for rf in results_file_dirs:
        print "getting results from '" + rf + "'..."

        if os.path.isfile(rf):
            results_files = [rf]
            all_test_oidxs = [test_oidxs]
            all_predicates_to_evaluate = [predicates_to_evaluate]
        else:
            results_files = []
            all_test_oidxs = []
            all_predicates_to_evaluate = []
            for root, dirs, files in os.walk(rf):
                for fn in files:
                    if fn == target_fn:
                        results_files.append(os.path.join(root, fn))

                        # For cross validation, set test fold based on parent directory.
                        if test_fold == -1:
                            curr_test = int(root.split('/')[-1])
                            test_oidxs = folds[curr_test]
                            predicates_to_evaluate = []
                            for pidx in range(0, len(predicates)):
                                num_pos = sum([1 for oidx in range(0, 32)
                                               if oidx not in test_oidxs and labels[oidx][pidx] == 1])
                                num_neg = sum([1 for oidx in range(0, 32)
                                               if oidx not in test_oidxs and labels[oidx][pidx] == 0])
                                if num_pos >= svm_examples and num_neg >= svm_examples:
                                    predicates_to_evaluate.append(pidx)
                            all_test_oidxs.append(test_oidxs)
                            all_predicates_to_evaluate.append(predicates_to_evaluate)

        avg_ms = []
        avg_ss = []
        avg_as = []
        avg_ps = []
        avg_rs = []
        avg_fs = []
        pred_cms = [[[0, 0], [0, 0]] for _ in range(len(predicates))]
        for ridx in range(len(results_files)):
            results_file = results_files[ridx]
            test_oidxs = all_test_oidxs[ridx]
            predicates_to_evaluate = all_predicates_to_evaluate[ridx]
            if verbose:
                print "... getting results from file '" + results_file + "'..."

            with open(results_file, 'rb') as f:
                results = pickle.load(f)

            # Calculate mean squared error.
            # MSE should ignore 0.5 labels.
            mse = {}
            for oidx in test_oidxs:
                mask = [pidx for pidx in range(len(labels[oidx])) if labels[oidx][pidx] == 0 or labels[oidx][pidx] == 1]
                mse[oidx] = mean_squared_error([labels[oidx][pidx] for pidx in mask if pidx in predicates_to_evaluate],
                                               [results[oidx][pidx] for pidx in mask if pidx in predicates_to_evaluate])
            avg_mse = np.mean([mse[oidx] for oidx in test_oidxs])

            # Calculate confusion matrices to get precision, recall, and f1.
            # Make majority class assumption for 0.5 ratings (negative label)
            s = {}
            a = {}
            p = {}
            r = {}
            f1 = {}
            pos = {}  # by oidx, the predicates predicted positive
            neg = {}  # by oidx, the predicates predicted negative
            for oidx in test_oidxs:
                cm = [[0, 0], [0, 0]]
                cm_strs = [[[], []], [[], []]]
                pos[oidx] = []
                neg[oidx] = []
                for pidx in predicates_to_evaluate:
                    if labels[oidx][pidx] == 0 or labels[oidx][pidx] == 1:
                        d = 1 if results[oidx][pidx] > 0.5 else 0
                        cm[labels[oidx][pidx]][d] += 1
                        cm_strs[labels[oidx][pidx]][d].append(predicates[pidx])
                        pred_cms[pidx][labels[oidx][pidx]][d] += 1
                        if d == 1:
                            pos[oidx].append(pidx)
                        elif d == 0:
                            neg[oidx].append(pidx)

                if verbose > 0:
                    print "object " + str(oidx)
                    print "\ttp: " + str(cm_strs[1][1])
                    print "\tfp: " + str(cm_strs[0][1])
                    print "\tfn: " + str(cm_strs[1][0])
                    print "\ttn: " + str(cm_strs[0][0])

                _p, _r, _f1 = get_p_r_f1(cm)
                s[oidx] = cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]
                a[oidx] = (cm[0][0] + cm[1][1]) / float(s[oidx])
                p[oidx] = _p
                r[oidx] = _r
                f1[oidx] = _f1

            if verbose > 0:
                for idx in range(len(test_oidxs)):
                    for jdx in range(idx + 1, len(test_oidxs)):
                        print ("pos " + str(test_oidxs[idx]) + ", neg " + str(test_oidxs[jdx]) + ": " +
                               str([predicates[pidx] for pidx in pos[test_oidxs[idx]]
                                    if pidx in neg[test_oidxs[jdx]]]))
                        print ("neg " + str(test_oidxs[idx]) + ", pos " + str(test_oidxs[jdx]) + ": " +
                               str([predicates[pidx] for pidx in neg[test_oidxs[idx]]
                                    if pidx in pos[test_oidxs[jdx]]]))

            avg_s = np.mean([s[oidx] for oidx in s])
            avg_a = np.mean([a[oidx] for oidx in a])
            avg_p = np.mean([p[oidx] for oidx in p])
            avg_r = np.mean([r[oidx] for oidx in r])
            avg_f1 = np.mean([f1[oidx] for oidx in f1])

            avg_ms.append(avg_mse)
            avg_ss.append(avg_s)
            avg_as.append(avg_a)
            avg_ps.append(avg_p)
            avg_rs.append(avg_r)
            avg_fs.append(avg_f1)

        # Print info.
        print "... avg labels:\t" + str(np.mean(avg_ss)) + "\t+/- " + str(np.std(avg_ss))
        print "... average mse:\t" + str(np.mean(avg_ms)) + "\t+/- " + str(np.std(avg_ms))
        print "... average acc:\t" + str(np.mean(avg_as)) + "\t+/- " + str(np.std(avg_as))
        print "... average p:\t" + str(np.mean(avg_ps)) + "\t+/- " + str(np.std(avg_ps))
        print "... average r:\t" + str(np.mean(avg_rs)) + "\t+/- " + str(np.std(avg_rs))
        print "... average f:\t" + str(np.mean(avg_fs)) + "\t+/- " + str(np.std(avg_fs))

        prf = [get_p_r_f1(pred_cms[pidx]) for pidx in range(len(predicates))]
        pp = [prf[pidx][0] for pidx in predicates_to_evaluate]
        pr = [prf[pidx][1] for pidx in predicates_to_evaluate]
        pf = [prf[pidx][2] for pidx in predicates_to_evaluate]
        print "... average pred p:\t" + str(np.mean(pp)) + "\t+/- " + str(np.std(pp))
        print "... average pred r:\t" + str(np.mean(pr)) + "\t+/- " + str(np.std(pr))
        print "... average pred f:\t" + str(np.mean(pf)) + "\t+/- " + str(np.std(pf))

        rfr[rf] = {"mse": avg_ms, "acc": avg_as,
                   "op": avg_ps, "or": avg_rs, "of": avg_fs,
                   "pp": pp, "pr": pr, "pf": pf}

        print "... done"

    # Do statistical tests.
    print "running statistical tests..."
    for idx in range(len(rfr.keys())):
        for jdx in range(idx + 1, len(rfr.keys())):
            for metric in ["acc", "op", "or", "of", "pp", "pr", "pf"]:
                t, p = ttest_ind(rfr[rfr.keys()[idx]][metric], rfr[rfr.keys()[jdx]][metric])
                if p < 0.05:
                    print "\t" + "\t".join([rfr.keys()[idx], rfr.keys()[jdx], metric, str(p)])
    print "... done"

    # Examine predicate performance
    if verbose > 1:
        print "examining predicate performance..."
        for rfdidx in range(len(results_file_dirs)):
            for rfdjdx in range(rfdidx + 1, len(results_file_dirs)):
                print results_file_dirs[rfdidx], results_file_dirs[rfdjdx]
                f1_diffs = {predicates[predicates_to_evaluate[idx]]:
                            rfr[results_file_dirs[rfdidx]]["pf"][idx] - rfr[results_file_dirs[rfdjdx]]["pf"][idx]
                            for idx in range(len(predicates_to_evaluate))}
                for key, value in sorted(f1_diffs.items(), key=operator.itemgetter(1)):
                    if abs(value) > 0:
                        print ('\t' + key + ": " + str(value) + " (" +
                               str(rfr[results_file_dirs[rfdidx]]["pf"][predicates_to_evaluate.index(
                                   predicates.index(key))]) + " - " +
                               str(rfr[results_file_dirs[rfdjdx]]["pf"][predicates_to_evaluate.index(
                                   predicates.index(key))]) + ")")
        print "... done"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True,
                        help="data directory")
    parser.add_argument('--test_fold', type=int, required=True,
                        help="fold on which to test classifiers; if -1, induce from directory structure")
    parser.add_argument('--results_file', type=str, required=True,
                        help="pickle of result or directory with single-layer subs with results pickles")
    parser.add_argument('--svm_examples', type=int, required=True,
                        help="how many pos+neg examples before evaluation continues")
    parser.add_argument('--alternative_labels', type=str, required=False,
                        help="specify labels pickle; defaults to expected location otherwise")
    parser.add_argument('--behavior', type=str, required=False,
                        help="get results for a particular behavior only when doing directory search")
    parser.add_argument('--verbose', type=int, required=False,
                        help="how much detail to show")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
