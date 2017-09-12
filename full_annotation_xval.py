#!/usr/bin/env python
__author__ = 'jesse'

import argparse
import os
import pickle
import time
from functions import get_data_for_classifier, get_signed_kappa, get_labels


def main():

    # Set the behaviors and modalities to consider.
    modalities = ["fpfh", "fc7", "haptics"]  # Always use these.

    # SURF modalities
    # behaviors = ["drop", "grasp", "hold", "lift", "look", "lower", "press", "push"]
    # for b in behaviors:  # SURF
    #     modalities.append(b + "_surf")  # SURF

    # Color alternatives
    modalities.append("color")  # I, Spy era
    # modalities.extend(["rgb", "hsv"])  # Alternatives

    # Audio alternatives
    modalities.append("audio")  # I, Spy era
    # modalities.append("corl_audio")  # Alternative

    nb_objects = 32

    # Convert flags to local variables.
    indir = FLAGS_indir
    kernel = FLAGS_kernel
    outfile = FLAGS_outfile

    print "reading in folds, labels, predicates, and features..."
    labels = get_labels(indir, os.path.join(indir, 'full_annotations.pickle'))

    with open(os.path.join(indir, 'predicates.pickle'), 'rb') as f:
        predicates = pickle.load(f)
        # predicates = predicates[0:1]  # DEBUG
        nb_predicates = len(predicates)
    feature_fn = os.path.join(indir, 'features.pickle')
    with open(feature_fn, 'rb') as f:
        object_feats = pickle.load(f)

    contexts = []
    for oidx in range(nb_objects):
        contexts = []
        for b in object_feats[oidx]:
            for m in modalities:
                if m not in object_feats[oidx][b]:
                    continue
                contexts.append((b, m))
    contexts_str = '.'.join([b + ',' + m for b, m in contexts])

    # Leave out one object at a time, training on others, then testing each predicate's result.
    # Track confusion matrices on a per-predicate, not per-object, basis.
    cms = [[[0, 0], [0, 0]] for _ in range(nb_predicates)]
    trained_preds = []
    outstanding_jobs = []  # tuples of test_oidx, pidx
    print "launching jobs..."
    for test_oidx in range(nb_objects):

        # Fit SVMs.
        print "...test oidx " + str(test_oidx) + " fitting SVMs for all predicates against train objects..."
        for pidx in range(nb_predicates):
            train_pairs = [(oidx, labels[oidx][pidx])
                           for oidx in [oidx for oidx in range(nb_objects) if oidx != test_oidx]
                           if labels[oidx][pidx] == 0 or labels[oidx][pidx] == 1]
            if (len([oidx for oidx, l in train_pairs if l == 1]) > 0 and
                    len([oidx for oidx, l in train_pairs if l == 0]) > 0):
                print "...... launching '" + str(predicates[pidx]) + "' classifier..."
                if pidx not in trained_preds:
                    trained_preds.append(pidx)

                tfn = "full_annotation_xval_" + str(test_oidx) + "_" + str(pidx) + ".pickle"
                cmd = ("full_annotation_train.py" +
                       " --indir " + indir +
                       " --kernel " + kernel +
                       " --contexts " + contexts_str +
                       " --test_oidx " + str(test_oidx) +
                       " --pidx " + str(pidx) +
                       " --outfile " + tfn)
                os.system("condorify_cpu " + cmd + " " + tfn + ".log")
                outstanding_jobs.append((test_oidx, pidx))
            else:
                print "...... skipped '" + str(predicates[pidx]) + "' due to insufficient label diversity"
        print "...... done"
    print "... done"

    print "collecting jobs..."
    pkas = [None for _ in range(nb_predicates)]  # store kappas for reporting in CSV
    while len(outstanding_jobs) > 0:
        time.sleep(10)
        newly_finished = []
        for test_oidx, pidx in outstanding_jobs:
            try:
                tfn = "full_annotation_xval_" + str(test_oidx) + "_" + str(pidx) + ".pickle"
                with open(tfn, 'rb') as f:
                    cs, kas = pickle.load(f)
                    pkas[pidx] = kas

                    dec = 0  # track decision over contexts
                    for b, m in contexts:
                        # Use classifier on held-out object.
                        obs = get_data_for_classifier(b, m, [(test_oidx, labels[test_oidx][pidx])],
                                                      object_feats)[0]
                        ds = cs[b][m].predict(obs)
                        d = 1 if sum(ds) > 0 else -1
                        dec += d * kas[b][m]

                    # Record result in confusion matrix
                    fd = 1 if dec > 0 else 0
                    cms[pidx][labels[test_oidx][pidx]][fd] += 1

                    newly_finished.append((test_oidx, pidx))

                os.system("rm " + tfn)
                os.system("rm " + tfn + ".log")
                os.system("rm err." + tfn + ".log")
            except (IOError, ValueError):
                pass
        if len(newly_finished) > 0:
            print "... gathered " + str(len(newly_finished)) + " jobs"
        outstanding_jobs = [j for j in outstanding_jobs if j not in newly_finished]
        if len(newly_finished) > 0:
            print "... " + str(len(outstanding_jobs)) + " remain"

    # Calculate kappas with human labels for each predicate, reporting those and average.
    tk = 0
    print "predicate\tkappa"
    for pidx in trained_preds:
        ka = get_signed_kappa(cms[pidx])
        print predicates[pidx] + "\t" + str(ka)
        tk += ka
    tk /= len(trained_preds)
    print "average: " + str(tk)

    # Write outfile with this information
    with open(outfile, 'w') as f:
        f.write(','.join(["predicate", "kappa_with_labels"] + [b + '/' + m for b, m in contexts]) + '\n')
        f.write('\n'.join([','.join([predicates[pidx], str(get_signed_kappa(cms[pidx]))]
                                    + [str(pkas[pidx][b][m]) for b, m in contexts])
                          for pidx in trained_preds]) + '\n')
        f.write('\n' + "average: " + str(tk) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True,
                        help="data directory")
    parser.add_argument('--kernel', type=str, required=True,
                        help="SVM kernel to use (linear, poly, rbf)")
    parser.add_argument('--outfile', type=str, required=True,
                        help="text file to write results to")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
