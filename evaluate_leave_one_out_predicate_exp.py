#!/usr/bin/env python
__author__ = 'jesse'
''' Train mult-modal mahalanobis-based neighbor classifiers.
'''

import argparse
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import os
import pickle
import sys
from scipy.stats import ttest_ind


# Returns non-negative kappa.
def get_signed_kappa(cm):

    s = float(cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
    assert s > 0
    po = (cm[1][1] + cm[0][0]) / s
    ma = (cm[1][1] + cm[1][0]) / s
    mb = (cm[0][0] + cm[0][1]) / s
    pe = (ma + mb) / s
    kappa = (po - pe) / (1 - pe) if pe < 1 else 0
    return kappa


# Given a confusion matrix, return the precision, recall and f1.
def get_p_r_f1(cm):

    p = float(cm[1][1]) / (cm[1][1] + cm[0][1]) if (cm[1][1] + cm[0][1]) > 0 else 0
    r = float(cm[1][1]) / (cm[1][1] + cm[1][0]) if (cm[1][1] + cm[1][0]) > 0 else 0
    f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
    return p, r, f1


def get_labels(indir, alt_fn):
    with open(os.path.join(indir, "labels.pickle"), 'rb') as f:
        labels = pickle.load(f)
    if alt_fn is not None:
        with open(alt_fn, 'rb') as f:
            alt_l = pickle.load(f)
            for oidx in alt_l:
                labels[oidx] = alt_l[oidx]
    return labels


def main():

    nb_objects = 32
    expected_trials = 100  # 10

    # Convert flags to local variables.
    data_dir = FLAGS_data_dir
    results_dir = FLAGS_results_dir
    metrics = FLAGS_metrics.split(',')
    alternative_labels = FLAGS_alternative_labels
    preds = FLAGS_preds
    trange = FLAGS_range.split(',')

    # Headers to use for figure making
    headers = {"uniform": "random (uniform)",
               "cos_top3_kappa": "guided (lex)", # "prior_kappa": "prior",
               "ba": "guided (ba)", "kba": "guided (lex+ba)"}

    # Read in labels and decision results.
    labels = get_labels(data_dir, alternative_labels)
    with open(os.path.join(data_dir, 'predicates.pickle'), 'rb') as f:
        predicates = pickle.load(f)
        if preds is None:
            preds = predicates[:]
        else:
            preds = preds.split(',')
        nb_predicates = len(predicates)

    # Read in results per predicate, calculating num contexts, num behaviors, and performance.
    predicates_evaluated = []
    nb_train = 0
    nb_test = 0
    weights = None  # list of weighting scheme names
    thresholds = None  # list of independent variable adjusted during experiment
    nb_tr_c = {}  # weight, then list parallel to thresholds
    nb_tr_b = {}
    nb_tr_t = {}
    nb_te_c = {}
    nb_te_b = {}
    nb_te_t = {}
    avg_k = {}
    avg_f = {}
    avg_a = {}
    avg_kpt = {}
    pred_ks = {}  # kappas per pidx
    for pidx in range(nb_predicates):
        if predicates[pidx] not in preds:
            continue

        try:
            with open(os.path.join(results_dir, str(pidx) + ".pickle"), 'rb') as f:
                wns, tr_c_used, tr_b_used, tr_t_used, te_c_used, te_b_used, te_t_used, decisions = pickle.load(f)
                predicates_evaluated.append(pidx)
                nb_train += nb_objects - len(decisions[wns[0]][decisions[wns[0]].keys()[0]])
                nb_test += len(decisions[wns[0]][decisions[wns[0]].keys()[0]])
                pred_ks[pidx] = {}
                if weights is None:
                    weights = wns
                    thresholds = sorted(tr_c_used[weights[0]].keys())
                    for w in weights:
                        nb_tr_c[w] = [0 for _ in range(len(thresholds))]
                        nb_tr_b[w] = [0 for _ in range(len(thresholds))]
                        nb_tr_t[w] = [0 for _ in range(len(thresholds))]
                        nb_te_c[w] = [0 for _ in range(len(thresholds))]
                        nb_te_b[w] = [0 for _ in range(len(thresholds))]
                        nb_te_t[w] = [0 for _ in range(len(thresholds))]
                        avg_k[w] = [0 for _ in range(len(thresholds))]
                        avg_f[w] = [0 for _ in range(len(thresholds))]
                        avg_a[w] = [0 for _ in range(len(thresholds))]
                        avg_kpt[w] = [0 for _ in range(len(thresholds))]
                for w in weights:
                    pred_ks[pidx][w] = []
                    nb_tr_c[w] = [nb_tr_c[w][idx] +
                                  np.mean([np.mean(
                                      [len(tr_c_used[w][thresholds[idx]][oidx][trial])
                                       for trial in range(len(tr_c_used[w][thresholds[idx]][oidx]))])
                                      for oidx in tr_c_used[w][thresholds[idx]].keys()])
                                  for idx in range(len(thresholds))]
                    nb_tr_b[w] = [nb_tr_b[w][idx] +
                                  np.mean([np.mean(
                                      [len(tr_b_used[w][thresholds[idx]][oidx][trial])
                                       for trial in range(len(tr_b_used[w][thresholds[idx]][oidx]))])
                                      for oidx in tr_b_used[w][thresholds[idx]].keys()])
                                  for idx in range(len(thresholds))]
                    nb_tr_t[w] = [nb_tr_t[w][idx] +
                                  np.mean([np.mean(
                                      [tr_t_used[w][thresholds[idx]][oidx][trial]
                                       for trial in range(len(tr_t_used[w][thresholds[idx]][oidx]))])
                                      for oidx in tr_t_used[w][thresholds[idx]].keys()])
                                  for idx in range(len(thresholds))]
                    nb_te_c[w] = [nb_te_c[w][idx] +
                                  np.mean([np.mean(
                                      [len(te_c_used[w][thresholds[idx]][oidx][trial])
                                       for trial in range(len(te_c_used[w][thresholds[idx]][oidx]))])
                                      for oidx in te_c_used[w][thresholds[idx]].keys()])
                                  for idx in range(len(thresholds))]
                    nb_te_b[w] = [nb_te_b[w][idx] +
                                  np.mean([np.mean(
                                      [len(te_b_used[w][thresholds[idx]][oidx][trial])
                                       for trial in range(len(te_b_used[w][thresholds[idx]][oidx]))])
                                      for oidx in te_b_used[w][thresholds[idx]].keys()])
                                  for idx in range(len(thresholds))]
                    nb_te_t[w] = [nb_te_t[w][idx] +
                                  np.mean([np.mean(
                                      [te_t_used[w][thresholds[idx]][oidx][trial]
                                       for trial in range(len(te_t_used[w][thresholds[idx]][oidx]))])
                                      for oidx in te_t_used[w][thresholds[idx]].keys()])
                                  for idx in range(len(thresholds))]

                    # Gets the average agreement scores per object for each timestep.
                    for idx in range(len(thresholds)):
                        nb_trials = expected_trials  # WARNING: specific to current experimental setup
                        tok = 0
                        tof = 0
                        toa = 0
                        tkpt = 0
                        trial_kappas = []
                        for trial in range(nb_trials):
                            cm = [[0, 0], [0, 0]]
                            for oidx in range(32):
                                if oidx in decisions[w][thresholds[idx]]:
                                    d = decisions[w][thresholds[idx]][oidx][trial]
                                    cm[labels[oidx][pidx]][d] += 1
                            ok = get_signed_kappa(cm)
                            _, _, of = get_p_r_f1(cm)
                            oa = (cm[0][0] + cm[1][1]) / float(cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
                            tok += ok
                            tof += of
                            toa += oa
                            tkpt += ok / nb_te_t[w][idx]
                            trial_kappas.append(ok)
                        avg_k[w][idx] += tok / nb_trials
                        avg_f[w][idx] += tof / nb_trials
                        avg_a[w][idx] += toa / nb_trials
                        avg_kpt[w][idx] += tkpt / nb_trials
                        pred_ks[pidx][w].append(trial_kappas)
        except IOError:
            pass

    # Get averages.
    nb_train /= float(len(predicates_evaluated))
    nb_test /= float(len(predicates_evaluated))
    for w in weights:
        nb_tr_c[w] = [nb_tr_c[w][idx] / float(len(predicates_evaluated)) for idx in range(len(thresholds))]
        nb_tr_b[w] = [nb_tr_b[w][idx] / float(len(predicates_evaluated)) for idx in range(len(thresholds))]
        nb_tr_t[w] = [nb_tr_t[w][idx] / float(len(predicates_evaluated)) for idx in range(len(thresholds))]
        nb_te_c[w] = [nb_te_c[w][idx] / float(len(predicates_evaluated)) for idx in range(len(thresholds))]
        nb_te_b[w] = [nb_te_b[w][idx] / float(len(predicates_evaluated)) for idx in range(len(thresholds))]
        nb_te_t[w] = [nb_te_t[w][idx] / float(len(predicates_evaluated)) for idx in range(len(thresholds))]
        avg_k[w] = [avg_k[w][idx] / float(len(predicates_evaluated)) for idx in range(len(thresholds))]
        avg_f[w] = [avg_f[w][idx] / float(len(predicates_evaluated)) for idx in range(len(thresholds))]
        avg_a[w] = [avg_a[w][idx] / float(len(predicates_evaluated)) for idx in range(len(thresholds))]
        avg_kpt[w] = [avg_kpt[w][idx] / float(len(predicates_evaluated)) for idx in range(len(thresholds))]
    print "evaluated " + str(len(predicates_evaluated)) + " predicates"
    print "predicates trained on an average of " + str(nb_train) + " objects and tested on " + str(nb_test)

    # Display results.
    if trange[0] == '':
        trange[0] = 0
    elif trange[1] == '':
        trange[1] = len(thresholds)
    trange = [int(t) for t in trange]
    for metric in metrics:
        print "metric: " + metric
        if metric == "trc":
            m = nb_tr_c
        elif metric == "trb":
            m = nb_tr_b
        elif metric == "trt":
            m = nb_tr_t
        elif metric == "tec":
            m = nb_te_c
        elif metric == "teb":
            m = nb_te_b
        elif metric == "tet":
            m = nb_te_t
        elif metric == "k":
            m = avg_k

            # calculate majority class 'no' baseline and add it to 'k' graph
            no = 0
            yes = 0
            for pidx in predicates_evaluated:
                if predicates[pidx] not in preds:
                    continue
                cm_no = [[0, 0], [0, 0]]
                cm_yes = [[0, 0], [0, 0]]
                for oidx in range(32):
                    if labels[oidx][pidx] == 1 or labels[oidx][pidx] == 0:
                        cm_no[labels[oidx][pidx]][0] += 1
                        cm_yes[labels[oidx][pidx]][1] += 1
                no += get_signed_kappa(cm_no)
                yes += get_signed_kappa(cm_yes)
            # m['no'] = [no / float(len(predicates_evaluated)) for _ in range(len(thresholds))]
            # m['yes'] = [yes / float(len(predicates_evaluated)) for _ in range(len(thresholds))]

            # Perform statistic tests between weighting schemes.
            print "statistical test results:"
            for idx in range(trange[0], trange[1]):
                print "\tsample at time " + str(thresholds[idx])
                for widx in range(len(m.keys())):
                    wi = m.keys()[widx]
                    for wjdx in range(widx + 1, len(m.keys())):
                        wj = m.keys()[wjdx]
                        print "\t\t" + wi + " against " + wj
                        rel_preds = []
                        for pidx in pred_ks.keys():
                            t, p = ttest_ind(pred_ks[pidx][wi][idx],
                                             pred_ks[pidx][wj][idx])
                            if p <= 0.05:
                                rel_preds.append((pidx, p))
                        if len(rel_preds) > 0:
                            print '\n\t\t\t' + '\n\t\t\t'.join([predicates[pidx] + " (" + str(p) + ")"
                                                                for pidx, p in rel_preds])

            # Find the predicates for this weight
            print "preds for which weight performance kappa exceeds uniform:"
            for w in m.keys():
                if w != 'uniform':
                    preds = []
                    for pidx in pred_ks.keys():
                        if (sum([sum([tk for tk in tks]) for tks in pred_ks[pidx][w][trange[0]:trange[1]]]) >
                                sum([sum([tk for tk in tks]) for tks in pred_ks[pidx]['uniform'][trange[0]:trange[1]]])):
                            preds.append(predicates[pidx])
                    print '\t', w, ','.join(preds)

        elif metric == "f":
            m = avg_f
        elif metric == "a":
            m = avg_a
        elif metric == "kpt":
            m = avg_kpt
        else:
            sys.exit("Unrecognized metric")
        print "average metric result:"
        legend = []
        for w in m.keys():
            if headers is None or w in headers.keys():
                print "\t" + w + ": " + str(sum(m[w]) / len(m[w]))  # average
                plt.plot(thresholds[trange[0]:trange[1]], m[w][trange[0]:trange[1]])
                legend.append(headers[w])
        plt.legend(legend)
        plt.ylabel(metric)
        plt.xlabel("allowed time (s)")
        plt.savefig(os.path.join(results_dir, metric + ".pdf"), bbox_inches='tight')
        plt.show()

    # Show kappa plot with error bars.
    err_k = {}
    legend_handles = []
    ax = plt.figure().gca()
    ax.set_xticks(thresholds[trange[0]:trange[1]])
    # ax.set_yticks(np.arange(0.668, 0.688, 0.005))  # Average
    # ax.set_yticks(np.arange(0.38, 0.43, 0.01))  # 'red'
    # ax.set_yticks(np.arange(0.668, 0.688, 0.005))  # 'full'
    markers = ['o', 'v', 's', 'D']
    colors = ['b', 'r', 'g', 'k']
    midx = 0
    for w in avg_k.keys():
        if headers is None or w in headers.keys():
            err_k[w] = [np.mean([np.std([pred_ks[pidx][w][idx][t] for t in range(expected_trials)])
                                 for pidx in pred_ks.keys()]) for idx in range(len(thresholds))]
            plt.plot(thresholds[trange[0]:trange[1]], avg_k[w][trange[0]:trange[1]],
                     markers[midx] + colors[midx] + '-')
            plt.plot(thresholds[trange[0]:trange[1]],
                     np.sum([avg_k[w][trange[0]:trange[1]], err_k[w][trange[0]:trange[1]]], axis=0),
                     colors[midx] + '--')
            plt.plot(thresholds[trange[0]:trange[1]],
                     np.sum([avg_k[w][trange[0]:trange[1]], np.negative(err_k[w][trange[0]:trange[1]])], axis=0),
                     colors[midx] + '--')

            ll = mlines.Line2D([], [], color=colors[midx], marker=markers[midx],
                               markersize=10, label=headers[w])
            legend_handles.append(ll)

            midx += 1
    plt.legend(handles=legend_handles)

    for label in (plt.subplot().get_xticklabels() + plt.subplot().get_yticklabels()):
        label.set_fontsize(18)

    # plt.ylim([y_min, y_max])
    # plt.xlim([x_min, x_max])

    plt.ylabel("Recognition Performance (kappa)", fontsize=23)
    plt.xlabel("Exploration Budget (seconds)", fontsize=23)
    plt.legend(loc=4, numpoints=1, fontsize=25)
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "error_bars_grid.pdf"), bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help="data directory")
    parser.add_argument('--results_dir', type=str, required=True,
                        help="the object id held out for testing")
    parser.add_argument('--metrics', type=str, required=True,
                        help="the metrics to plot")
    parser.add_argument('--alternative_labels', type=str, required=False,
                        help="specify labels pickle; labels in this pickle will override defaults")
    parser.add_argument('--preds', type=str, required=False,
                        help="specify the predicates to gather info from")
    parser.add_argument('--range', type=str, required=False,
                        help="comma-separated threshold range indices; blank spans")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
