#!/usr/bin/env python
__author__ = 'jesse'
''' Train mult-modal mahalanobis-based neighbor classifiers.
'''

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from functions import get_p_r_f1, get_kappa


def main():

    nb_objects = 32
    behavior_time = {"drop": 10, "grasp": 5, "hold": 1, "lift": 10, "look": 1, "lower": 10, "press": 5, "push": 5}

    # Convert flags to local variables.
    data_dir = FLAGS_data_dir
    results_dir = FLAGS_results_dir
    weightings = FLAGS_weightings.split(',')
    metric = FLAGS_metric

    # Read in labels and decision results.
    with open(os.path.join(data_dir, 'labels.pickle'), 'rb') as f:
        labels = pickle.load(f)
    with open(os.path.join(data_dir, 'predicates.pickle'), 'rb') as f:
        predicates = pickle.load(f)
        nb_predicates = len(predicates)
    oidxs = range(nb_objects)
    decision_matrices = []
    behavior_matrices = []
    self_est_kappa_decs = []
    self_est_kappa_bs = []
    for oidx in oidxs:
        with open(os.path.join(results_dir, str(oidx) + '.pickle'), 'rb') as f:
            dm, sekd, bm, sekb = pickle.load(f)
            decision_matrices.append(dm)
            self_est_kappa_decs.append(sekd)
            behavior_matrices.append(bm)
            self_est_kappa_bs.append(sekb)

    # Calculate per-predicate performance for kappa-only and varying requirements for training examples.
    p = {}  # indexed by "self" or an integer representing the required minimum training examples to trust self
    r = {}
    f = {}
    ka = {}
    ac = {}
    nb = {}
    bt = {}
    npb = {}
    # conditions = ["pos", "neg", "self"]
    conditions = ["self"]
    conditions.extend(decision_matrices[0].keys())
    for w in weightings:
        p[w] = {}
        r[w] = {}
        f[w] = {}
        ka[w] = {}
        ac[w] = {}
        nb[w] = {}
        bt[w] = {}
        npb[w] = {}
        for cond in conditions:
            ps = []
            rs = []
            fs = []
            kas = []
            acs = []
            nbs = []
            bts = []
            npbs = []
            for pidx in range(nb_predicates):
                cm = [[0, 0], [0, 0]]  # confusion matrix for this predicate's labels across all objects
                _nb = _bt = _npb = nl = 0
                for oidx in range(nb_objects):
                    if labels[oidx][pidx] == 0 or labels[oidx][pidx] == 1:
                        nl += 1
                        if cond == "pos":
                            d = 1
                            bs = []
                        elif cond == "neg":
                            d = 0
                            bs = []
                        elif cond == "self":
                            d = 1 if self_est_kappa_decs[oidx][pidx] > 0.5 else 0
                            bs = self_est_kappa_bs[oidx][pidx]
                        else:
                            d = 1 if decision_matrices[oidx][cond][w][pidx] > 0.5 else 0
                            bs = behavior_matrices[oidx][cond][w][pidx]
                        cm[labels[oidx][pidx]][d] += 1
                        _nb += len(bs)
                        _bt += sum([behavior_time[b] for b in bs])

                        holding_ob = False
                        ob_up = False
                        for b in bs:
                            _npb += 1  # the behavior we need to perform
                            if not holding_ob and b in ["drop", "hold", "lift", "lower"]:
                                _npb += 1  # grasp
                                holding_ob = True
                                if not ob_up and b == "lower":
                                    _npb += 1  # lift
                                if ob_up and b in ["drop", "lift"]:
                                    _npb += 1  # lower
                                if b == "lift":
                                    ob_up = True
                                if b == "drop":
                                    holding_ob = False
                                    ob_up = False
                                continue
                            if holding_ob and b in ["grasp", "look", "press", "push"]:
                                _npb += 1  # drop
                                if ob_up:
                                    _npb += 1  # lower
                                ob_up = False
                                holding_ob = False
                                if b == "grasp":
                                    holding_ob = True

                _nb /= float(nl)
                _bt /= float(nl)
                _npb /= float(nl)
                _p, _r, _f = get_p_r_f1(cm)
                _k = get_kappa(cm)
                _a = (cm[0][0] + cm[1][1]) / float(cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
                ps.append(_p)
                rs.append(_r)
                fs.append(_f)
                kas.append(_k)
                acs.append(_a)
                nbs.append(_nb)
                bts.append(_bt)
                npbs.append(_npb)
            p[w][cond] = np.mean(ps)
            r[w][cond] = np.mean(rs)
            f[w][cond] = np.mean(fs)
            ka[w][cond] = np.mean(kas)
            ac[w][cond] = np.mean(acs)
            nb[w][cond] = np.mean(nbs)
            bt[w][cond] = np.mean(bts)
            npb[w][cond] = np.mean(npbs)

    # Display results.
    for w in weightings:
        print str(w) + ":"
        for cond in conditions:
            print "\t" + str(cond) + ":"
            print "\t\tp:\t" + str(p[w][cond])
            print "\t\tr:\t" + str(r[w][cond])
            print "\t\tf:\t" + str(f[w][cond])
            print "\t\tk:\t" + str(ka[w][cond])
            print "\t\ta:\t" + str(ac[w][cond])
            print "\t\tnb:\t" + str(nb[w][cond])
            print "\t\tbt:\t" + str(bt[w][cond])
            print "\t\tnpb:\t" + str(npb[w][cond])

    # Plot results.
    m = None
    if metric == "p":
        m = p
    elif metric == "r":
        m = r
    elif metric == "f":
        m = f
    elif metric == "k":
        m = ka
    elif metric == "a":
        m = ac
    elif metric == "nb":
        m = nb
    elif metric == "bt":
        m = bt
    elif metric == "npb":
        m = npb
    x = [c for c in conditions if type(c) is int]
    for w in weightings:
        y = [m[w][cond] for cond in x]
        plt.plot(x, y)
    legend = weightings[:]
    for c in [c for c in conditions if type(c) is str]:
        y = [m[weightings[0]][c] for _ in range(len(x))]
        plt.plot(x, y)
        legend.append(c)
    plt.legend(legend)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help="data directory")
    parser.add_argument('--results_dir', type=str, required=True,
                        help="the object id held out for testing")
    parser.add_argument('--weightings', type=str, required=True,
                        help="the weighting schemes to consider")
    parser.add_argument('--metric', type=str, required=True,
                        help="the metric to plot")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
