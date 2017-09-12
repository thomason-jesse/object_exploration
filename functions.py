import keras.backend as K
import copy
import operator
import os
import pickle
import numpy as np
import random
import sys
from keras.preprocessing.sequence import pad_sequences
from sklearn.svm import SVC


picture_size = 224  # standard resize in vision community for picture cropping/extraction.


# Return the mse over only predicates that are labeled.
# TODO: adjust error per class by 1/num_labels so propagation doesn't disproportionately move those w more labels.
# TODO: looks like this might be achieved with the class_weight parameter to fit()
def label_only_mse(y_true, y_pred):
    unlabeled_mask = 2 * K.abs(y_true - 0.5)  # 1 for labelled, 0 otherwise
    y_true_wide = 2 * (y_true - 0.5)  # 0 for unlabelled, -1 for negative, 1 for positive
    y_pred_wide = 2 * (y_pred - 0.5)
    return K.mean(K.square((y_pred_wide * unlabeled_mask) - y_true_wide), axis=-1)


# Takes in y_true, y_pred Tensors and returns a singleton tensor representing the f1 measure of
# the label agreement. Labels in y_true with value 0.5 are ignored since these in the original
# data are object/predicate pairs for which we have no information or consensus.
def pred_f1(y_true, y_pred):
    predicted = K.round(y_pred)  # Since keras uses round to even, 0.5 -> 0 which we want.

    unlabeled_mask = 2 * K.abs(y_true - 0.5)  # 1 for labelled, 0 otherwise

    tp = K.tf.count_nonzero(predicted * y_true * unlabeled_mask)
    fp = K.tf.count_nonzero(predicted * (y_true - 1) * unlabeled_mask)
    fn = K.tf.count_nonzero((predicted - 1) * y_true * unlabeled_mask)

    p_dem_nonzero = K.tf.count_nonzero(tp + fp)  # 1 if tp + fp > 0, else 0
    # multiplying the numerator by this zeros it if tp + fp == 0, has no effect if tp + fp > 0
    # adding (1 - this) to denominator causes 0 denominator to become 1, no effect if tp + fp > 0
    # So division by zero becomes 1/0, other types of division unaffected.
    precision = K.tf.divide(tp * p_dem_nonzero, tp + fp + 1 - p_dem_nonzero)
    r_dem_nonzero = K.tf.count_nonzero(tp + fn)
    recall = K.tf.divide(tp * r_dem_nonzero, tp + fn + 1 - r_dem_nonzero)
    f_dem_nonzero = K.tf.count_nonzero(precision + recall)
    f_dem_nonzero = K.tf.cast(f_dem_nonzero, K.tf.float64)
    fmeasure = K.tf.div(2 * precision * recall * f_dem_nonzero, precision + recall + 1 - f_dem_nonzero)

    return fmeasure


def pred_acc(y_true, y_pred):
    predicted = K.round(y_pred)  # Since keras uses round to even, 0.5 -> 0 which we want.

    unlabeled_mask = 2 * K.abs(y_true - 0.5)  # 1 for labelled, 0 otherwise

    tp = K.tf.count_nonzero(predicted * y_true * unlabeled_mask)
    tn = K.tf.count_nonzero((predicted - 1) * (y_true - 1) * unlabeled_mask)
    fp = K.tf.count_nonzero(predicted * (y_true - 1) * unlabeled_mask)
    fn = K.tf.count_nonzero((predicted - 1) * y_true * unlabeled_mask)

    acc = K.tf.divide(tp + tn, tp + tn + fp + fn)
    return acc


def get_input_output_data_for_oidx(oidx, indir, longest_times, num_features,
                                   labels, behaviors, modalities,
                                   obs_to_get, predict_properties, verbose):

    if verbose == 2:
        print "get_input_output_data_for_oidx retrieving for " + str(oidx) + "..."
    with open(os.path.join(indir, str(oidx)+".pickle"), 'rb') as f:
        ind = pickle.load(f)
    xl = []
    for idx in range(0, len(behaviors)):
        b = behaviors[idx]
        for m in modalities[idx]:
            x = ind[b][m]
            x = [x[jdx] for jdx in obs_to_get]
            if m == "vision2D":
                x = np.reshape(x, (len(x), picture_size, picture_size, 3))
            elif m == "visionVGG" or m == "audio_ispy" or m == "haptic_ispy" or m == "color" or m == "fpfh":
                x = np.reshape(x, (len(x), num_features[b][m]))  # flat feature vector, single observation.
            else:
                x = pad_sequences(x, padding='post', maxlen=longest_times[b][m], value=0)
                x = x.reshape(x.shape[0], longest_times[b][m], num_features[b][m], 1)
            xl.append(x)
    if predict_properties:
        y = np.asarray([labels[oidx]] * xl[0].shape[0])  # Same labels for this object for all inputs.
    else:
        y = [oidx] * xl[0].shape[0]  # oidx is label if predicting object itself.
        y = np.expand_dims(y, -1)
    if len(xl) == 1:
        xl = xl[0]  # un-nest is single model input.
    if verbose == 2:
        print "... get_input_output_data_for_oidx done"
    return xl, y


def get_data_for_model(oidxs, batch_size, indir, longest_times, num_features,
                       labels, behaviors, modalities,
                       obs_to_get,  predict_properties, verbose):

    if verbose == 2:
        print "get_data_for_model running forever..."
    r = 0
    while True:
        random.shuffle(oidxs)
        for oidx in oidxs:
            oidx_in, oidx_out = get_input_output_data_for_oidx(oidx, indir, longest_times, num_features,
                                                               labels, behaviors, modalities,
                                                               obs_to_get, predict_properties, verbose)
            c = 0
            while c < len(oidx_out):
                m = min(batch_size - r, len(oidx_out) - c)
                if verbose == 2:
                    print "... get_data_for_model yielding " + str(m-c) + " more pairs"
                if len(behaviors) == 1 and len(modalities) == 1:
                    yield oidx_in[c:m], oidx_out[c:m]
                else:
                    xr = [oidx_in[cidx][c:m] for cidx in range(len(oidx_in))]
                    yield xr, oidx_out[c:m]
                c = m
                r += m
                if r >= batch_size:
                    r = 0


def get_labels(indir, alt_fn):
    with open(os.path.join(indir, "labels.pickle"), 'rb') as f:
        labels = pickle.load(f)
    if alt_fn is not None:
        with open(alt_fn, 'rb') as f:
            alt_l = pickle.load(f)
            for oidx in alt_l:
                labels[oidx] = alt_l[oidx]
    return labels


def get_decisions_from_weights(ds, ws, contexts):
    r = {}
    bs = {}
    for pidx in ds:

        behavior_weights = {b: sum([ws[pidx][contexts.index((b, m))]
                                    for _b, m in contexts if _b == b])
                            for b, _ in contexts}
        p = n = 0
        _bs = []
        for b, w in sorted(behavior_weights.items(), key=operator.itemgetter(1)):
            d = sum([(2 * (ds[pidx][b][m] - 0.5)) * ws[pidx][contexts.index((b, m))]
                    for _b, m in contexts if _b == b])
            if d > 0:
                p += d
            else:
                n -= d
            _bs.append(b)
            if p > 0.5 or n > 0.5:
                break
        r[pidx] = 1 if (p > n and p > 0) else 0
        bs[pidx] = _bs

    return r, bs


def get_data_for_classifier(behavior, modality, pairs, object_feats, obs_limit=None):
    x = []
    y = []
    for oidx, label in pairs:
        nb_oidx_obs = 0
        for obs in object_feats[oidx][behavior][modality]:
            if obs_limit is None or nb_oidx_obs < obs_limit:
                x.append(obs)
                l = 1 if label == 1 else -1
                y.append(l)
                nb_oidx_obs += 1
    return x, y


def fit_classifier(kernel, behavior, modality, pairs, object_feats, obs_limit=None):
    x, y = get_data_for_classifier(behavior, modality, pairs, object_feats, obs_limit=obs_limit)
    assert len(x) > 0  # there is data
    assert min(y) < max(y)  # there is more than one label
    c = SVC(kernel=kernel, degree=2)
    c.fit(x, y)
    return c


# Given an SVM and its training data, fit that training data, optionally retraining leaving
# one object out at a time.
def get_classifier_results(c, behavior, modality, pairs, object_feats, xval, kernel, obs_limit=None):
    if xval is None:
        x, y = get_data_for_classifier(behavior, modality, pairs, object_feats, obs_limit=obs_limit)
        z = c.predict(x)
    else:
        x = []
        y = []
        z = []
        oidxs = list(set([oidx for (oidx, l) in pairs]))
        if len(oidxs) > 1:
            for oidx in oidxs:
                # Train a new classifier without data from oidx.
                xval_pairs = [(ojdx, l) for (ojdx, l) in xval if ojdx != oidx]
                ls = list(set([l for ojdx, l in xval_pairs]))
                if len(ls) == 2:
                    xval_c = fit_classifier(kernel, behavior, modality, xval_pairs, object_feats, obs_limit=obs_limit)
                else:
                    xval_c = None

                # Evaluate new classifier on held out oidx data and record results.
                xval_pairs = [(ojdx, l) for (ojdx, l) in pairs if ojdx == oidx]
                _x, _y = get_data_for_classifier(behavior, modality, xval_pairs, object_feats, obs_limit=obs_limit)
                if xval_c is not None:
                    _z = xval_c.predict(_x)
                else:  # If insufficient data, vote the same label as the training data. If no training data, vote no.
                    _z = [1 if len(ls) > 0 and ls[0] == 1 else -1 for _ in range(len(_x))]
                x.extend(_x)
                y.extend(_y)
                z.extend(_z)
        else:
            x, y = get_data_for_classifier(behavior, modality, pairs, object_feats, obs_limit=obs_limit)
            z = [-1 for _ in range(len(x))]  # Single object, so guess majority class no.
    return x, y, z


# Returns non-negative kappa.
def get_kappa(cm):
    return max(0, get_signed_kappa(cm))


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


# Same as get_margin_kappa below, but returns intact cm as well.
# TODO: this is silly inefficient, modularity-wise
def get_margin_kappa_and_cm(c, behavior, modality, pairs, object_feats, xval=None, kernel=None, minc=(0, 0),
                            obs_limit=None):
    x, y, z = get_classifier_results(c, behavior, modality, pairs, object_feats, xval, kernel, obs_limit=obs_limit)
    cm = [[0, 0], [0, 0]]
    for idx in range(len(x)):
        cm[1 if y[idx] == 1 else 0][1 if z[idx] == 1 else 0] += 1
    return get_signed_kappa(cm) if get_signed_kappa(cm) > minc[0] else minc[1], cm


# Given an SVM c and its training data, calculate the agreement with gold labels according to kappa
# agreement statistic at the observation level.
def get_margin_kappa(c, behavior, modality, pairs, object_feats, xval=None, kernel=None, minc=(0, 0),
                     obs_limit=None):
    x, y, z = get_classifier_results(c, behavior, modality, pairs, object_feats, xval, kernel, obs_limit=obs_limit)
    cm = [[0, 0], [0, 0]]
    for idx in range(len(x)):
        cm[1 if y[idx] == 1 else 0][1 if z[idx] == 1 else 0] += 1
    return get_signed_kappa(cm) if get_signed_kappa(cm) > minc[0] else minc[1]


# Given an SVM c and its training data, calculate the agreement with gold labels according to f1
# agreement statistic at the observation level.
def get_margin_f1(c, behavior, modality, pairs, object_feats, xval=None, kernel=None, minc=(0, 0)):
    x, y, z = get_classifier_results(c, behavior, modality, pairs, object_feats, xval, kernel)
    cm = [[0, 0], [0, 0]]
    for idx in range(len(x)):
        cm[1 if y[idx] == 1 else 0][1 if z[idx] == 1 else 0] += 1
    _, _, f1 = get_p_r_f1(cm)
    return f1 if f1 > minc[0] else minc[1]


# Given an SVM c and its training data, calculate the agreement with gold labels according to accuracy
# at the observation level.
def get_margin_acc(c, behavior, modality, pairs, object_feats, xval=None, kernel=None, minc=(0, 0)):
    x, y, z = get_classifier_results(c, behavior, modality, pairs, object_feats, xval, kernel)
    cm = [[0, 0], [0, 0]]
    for idx in range(len(x)):
        cm[1 if y[idx] == 1 else 0][1 if z[idx] == 1 else 0] += 1
    a = float(cm[1][1] + cm[0][0]) / float(cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
    return a if a > minc[0] else minc[1]


# Given a confusion matrix, return the precision, recall and f1.
def get_p_r_f1(cm):

    p = float(cm[1][1]) / (cm[1][1] + cm[0][1]) if (cm[1][1] + cm[0][1]) > 0 else 0
    r = float(cm[1][1]) / (cm[1][1] + cm[1][0]) if (cm[1][1] + cm[1][0]) > 0 else 0
    f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
    return p, r, f1


# Given reward and cost functions r and c and a horizon l, calculate the best path through the graph
# of object states to maximize expected reward.
# if by_time is true, horizon describes the maximum time allowed to sequences.
def get_best_sequence_of_behaviors(behaviors, r, c, l, maximize_reward=True, max_samples=None,
                                   random_walk=False, degenerate=False):
    assert max_samples is None or random_walk
    # print behaviors, r, l, maximize_reward, max_samples  # DEBUG

    # proper transition function
    t = {b: {} for b in behaviors}  # transition function from states to states; if empty, behavior not performable
    for b, s, e in [("drop", (1, 0), (0, 0)),  # states describe 'held', 'raised' object
                    ("grasp", (0, 0), (1, 0)),
                    ("hold", (1, 1), (1, 1)),
                    ("lift", (1, 0), (1, 1)),
                    ("look", (0, 0), (0, 0)),
                    ("lower", (1, 1), (1, 0)),
                    ("press", (0, 0), (0, 0)),
                    ("push", (0, 0), (0, 0))]:
        if b in t.keys():
            t[b][s] = e

    # degenerate transition function
    if degenerate:
        t = {b: {} for b in behaviors}
        for b, s, e in [("drop", (0, 0), (0, 0)),
                        ("grasp", (0, 0), (0, 0)),
                        ("hold", (0, 0), (0, 0)),
                        ("lift", (0, 0), (0, 0)),
                        ("look", (0, 0), (0, 0)),
                        ("lower", (0, 0), (0, 0)),
                        ("press", (0, 0), (0, 0)),
                        ("push", (0, 0), (0, 0))]:
            if b in t.keys():
                t[b][s] = e

    nb_obs = 5  # artifact of the dataset
    epsilon = 0.001  # used when not front-loading reward to allow otherwise zero-reward actions to be taken

    # Create a dictionary of how many times each action has been seen on a path, initialized to zero for
    # useful paths and max times for 0 reward paths (to cut down search space)
    ue = {b: 0 for b in behaviors}
    if maximize_reward:  # mark useless edges
        v = {b: r[b] / float(c[b]) for b in behaviors}  # expected reward per cost
        for b in behaviors:
            if r[b] == 0:
                ue[b] = nb_obs
    else:
        v = {b: (epsilon + r[b]) / float(c[b]) for b in behaviors}  # expected reward with fixed epsilon bonus

    # Sample up to max_samples paths, or get all
    enum = enumerate_paths_from_state(behaviors, t, c, (0, 0), l, ue, nb_obs, v, random_walk=random_walk)
    paths = []
    next_path = enum.next()
    if not random_walk:
        while next_path is not None and (max_samples is None or len(paths) < max_samples):
            paths.append(next_path)
            next_path = enum.next()
            # print "sampled " + str(len(paths)) + " / " + str(max_samples)  # DEBUG
    else:
        while len(paths) < max_samples:
            while next_path is None:
                enum = enumerate_paths_from_state(behaviors, t, c, (0, 0), l, ue, nb_obs, v, random_walk=random_walk)
                next_path = enum.next()
            paths.append(next_path)
            next_path = enum.next()
            # print "sampled " + str(len(paths)) + " / " + str(max_samples)  # DEBUG
    # print ".. done sampling"  # DEBUG
    # print behaviors, r, l, maximize_reward, max_samples  # DEBUG
    # print "paths: " + '\n'.join([str(p) for p in paths])  # DEBUG
    # _ = raw_input()  # DEBUG

    # Calculate a mask for each path to offer 0 reward for already-seen actions.
    paths_r = [[1 if paths[idx][:jdx].count(paths[idx][jdx]) < nb_obs else 0
               for jdx in range(len(paths[idx]))] for idx in range(len(paths))]

    # Get the top-scoring paths. For score-maximizing strategy, weight by expected reward-per-cost
    # towards the front of the exploration cycle.
    if not maximize_reward:
        scores = [sum([v[paths[idx][jdx]] * paths_r[idx][jdx] for jdx in range(len(paths[idx]))])
                  for idx in range(len(paths))]
    else:
        scores = [sum([sum([v[paths[idx][kdx]] * paths_r[idx][kdx]
                           for kdx in range(jdx)])
                      for jdx in range(1, len(paths[idx]) + 1)])
                  for idx in range(len(paths))]

    max_paths = [i for i, x in enumerate(scores)
                 if np.isclose(x, np.max([scores[idx] for idx in range(len(paths))]))]
    # print "max paths: " + '\n'.join([str([paths[idx], scores[idx]]) for idx in max_paths])  # DEBUG
    # _ = raw_input()  # DEBUG

    if maximize_reward:  # now minimize time among score-tied paths
        costs = [sum([sum([c[paths[idx][kdx]] for kdx in range(jdx)])
                      for jdx in range(len(paths[idx]))])
                 for idx in max_paths]
    else:
        costs = [sum([c[paths[idx][jdx]] for jdx in range(len(paths[idx]))])
                 for idx in max_paths]
    min_costs = [i for i, x in enumerate(costs)
                 if np.isclose(x, np.min([costs[idx] for idx in range(len(max_paths))]))]

    to_return = [paths[max_paths[idx]] for idx in min_costs]
    for tr in to_return:  # validation
        if not is_valid_path(tr, (0, 0), t, c, l):
            sys.exit("ERROR: invalid path created: " + str(tr))
    return to_return


# Given a set of edges, a graph, a start state, edge weights, and a budget, enumerate all paths through
# the graph.
def enumerate_paths_from_state(edges, t, c, s, l, he, m, v, random_walk=False):
    # print s, l, he, vn, m  # DEBUG
    out = [e for e in edges if s in t[e].keys()]  # out edges from current node
    paths = []  # track of what this pass turns up; in cases where none, may be valid base case
    ce = []  # edges we can't take due to cost
    ve = []  # edges we can't take because they lead to vacuous nodes
    # print out  # DEBUG

    # Collect set of possible edges to take.
    pe = []  # possible edges to take next
    for e in out:
        if l < c[e]:  # we don't have the budget to take edge e
            ce.append(e)
            continue
        if he[e] == m:  # edge e exhausted; if parts of the graph are unreached then favor paths that visit those first
            ve.append(e)
            continue
        pe.append(e)
    # print pe  # DEBUG

    # Sample from these edges weighted by their known values, taking DFS next edges one at a time until exhausted.
    while len(pe) > 0:

        pes = float(sum([v[e] for e in pe]))
        pep = [v[e] / pes if pes > 0 else 1.0 / len(pe) for e in pe]
        idx = np.random.choice(range(len(pe)), 1, p=pep)[0]

        e = pe[idx]
        # we can take the edge, so extend possible paths with those that come from taking this edge
        p = copy.copy(he)
        p[e] += 1
        enum = enumerate_paths_from_state(edges, t, c, t[e][s], l - c[e], p, m, v, random_walk=random_walk)
        ext = enum.next()
        while ext is not None:
            paths.append([e] + ext)
            # print "... yielding " + str([e] + ext)  # DEBUG
            yield [e] + ext
            ext = enum.next()

        del pe[idx]

        # If we're just doing random walks, don't save this state, just move on after taking weighted edge.
        if random_walk:
            break

    # non-failure base cases
    # print paths, ce, ve  # DEBUG
    if len(paths) == 0:
        if len([e for e in out if e in ce and e not in ve]) > 0:
            # we found no paths because we're out of budget for every valuable edge
            # print "... yielding budget base"  # DEBUG
            yield []
        elif np.alltrue([he[e] == m for e in he.keys()]):
            # we have exhausted the action space and stayed under budget
            # print "... yielding exhausted sapce base"  # DEBUG
            yield []
        elif np.alltrue([he[e] == m for e in out]):
            # we are in a terminal node (possible for subsets of graph) and have nothing left to do
            # print "... yielding terminal node base"  # DEBUG
            yield []

    # print "... yielding None"  # DEBUG
    yield None


# Return whether a given path, transition function, and budget are valid.
def is_valid_path(p, s, t, c, l):
    for v in p:
        if s in t[v].keys():
            s = t[v][s]
            l -= c[v]
            if l < 0:
                return False
        else:
            return False
    return True


# Return a set of lists representing unique multi-sets in this list of lists.
def get_representatives_of_unique_sets(l):
    ss = []
    rl = []
    for idx in range(len(l)):
        s = set([(m, l[idx].count(m)) for m in set(l[idx])])
        if s not in ss:
            ss.append(s)
            rl.append(l[idx])
    return rl


# List contains
def list_contains(small, big):
    for i in xrange(len(big) - len(small) + 1):
        for j in xrange(len(small)):
            if big[i + j] != small[j]:
                break
        else:
            return i, i + len(small)
    return False, False
