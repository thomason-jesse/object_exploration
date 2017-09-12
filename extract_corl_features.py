#!/usr/bin/env python

import argparse
import numpy as np
import os
import pickle


def main():

    # Load parameters from command line.
    ispy_processed_pickle = FLAGS_ispy_processed_pickle
    ispy_raw = FLAGS_ispy_raw
    surf_dir = FLAGS_surf_dir
    alt_color_dir = FLAGS_alt_color_dir
    alt_audio_dir = FLAGS_alt_audio_dir
    outfile = FLAGS_outfile

    # Parameters we expect.
    object_ids = range(32)
    behaviors = ["drop", "grasp", "hold", "lift", "look", "lower", "press", "push"]

    # Other I, Spy features.
    print "reading in I, Spy data"
    with open(ispy_processed_pickle, 'rb') as f:
        ispy_features = pickle.load(f)
    print "... done"

    print "extracting features for each object..."
    object_feats = []
    for oidx in object_ids:
        print str(oidx + 1) + "/" + str(len(object_ids))

        behavior_obs = {b: {} for b in behaviors}

        # I, Spy features.
        print "... getting and storing I, Spy features"
        # skips context 'grasp'-'finger'
        for b in behaviors:
            if b != "look":
                for m in ["audio", "haptics"]:
                    behavior_obs[b][m] = ispy_features[oidx][b][m]
            else:
                for m in ["color", "fpfh"]:
                    behavior_obs[b][m] = ispy_features[oidx][b][m]

                behavior_obs[b]['fc7'] = []
                with open(os.path.join(ispy_raw, "obj"+str(oidx+1), "look", "fc7", "features.csv"), 'r') as f:
                    trial_data = f.read().strip().split('\n')
                    for td in trial_data:
                        behavior_obs[b]['fc7'].append([float(feat) for feat in td.split(',')[1:]])
        print "...... done"

        # SURF features.
        print "... getting and storing SURF features"
        for b in behaviors:
            with open(os.path.join(surf_dir, b + "_surf200-pca.csv"), 'r') as f:
                trials = []
                for line in f.readlines():
                    p = line.strip().split(',')
                    l_oidx = int(p[0]) - 1
                    if l_oidx == oidx:
                        trials.append([float(val) for val in p[2:]])
                behavior_obs[b][b + "_surf"] = trials
                print "...... wrote surf features for " + b  # DEBUG

        print "... getting and storing alternative color features"
        for m in ['hsv', 'rgb']:
            with open(os.path.join(alt_color_dir, "look_" + m + "norm8.txt"), 'r') as f:
                trials = []
                for line in f.readlines():
                    p = line.strip().split(',')
                    l_oidx = int(p[0]) - 1
                    if l_oidx == oidx:
                        trials.append([float(val) for val in p[2:]])
                behavior_obs['look'][m] = trials
                print "...... wrote " + m + " features"  # DEBUG

        print "... getting and storing alternative audio features"
        for b in behaviors:
            if b != 'look':
                with open(os.path.join(alt_audio_dir, b + "_audio.txt"), 'r') as f:
                    trials = []
                    for line in f.readlines():
                        p = line.strip().split(',')
                        l_oidx = int(p[0]) - 1
                        if l_oidx == oidx:
                            trials.append([float(val) for val in p[2:]])
                    behavior_obs[b]['corl_audio'] = trials
                    print "...... wrote " + b + " features"  # DEBUG

        object_feats.append(behavior_obs)
    print "... done"

    # Write out longest times.
    print "writing features to file..."
    with open(outfile, 'wb') as f:
        pickle.dump(object_feats, f)
    print "... done"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ispy_processed_pickle', type=str, required=True,
                        help="processed ispy features")
    parser.add_argument('--ispy_raw', type=str, required=True,
                        help="raw ispy feature directory from which we extract VGG")
    parser.add_argument('--surf_dir', type=str, required=True,
                        help="directory in which to locate SURF features")
    parser.add_argument('--alt_color_dir', type=str, required=True,
                        help="directory in which to locate new color features")
    parser.add_argument('--alt_audio_dir', type=str, required=True,
                        help="directory in which to locate new audio features")
    parser.add_argument('--outfile', type=str, required=True,
                        help="pickle to write final data pickle")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
