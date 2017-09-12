#!/usr/bin/env python

import argparse
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

'''Extracts data from flat files and stores it in more coherent pickles for haptic and audio features.
Does not perform image feature extraction, but instead writes a list of image paths to be processed
on-the-fly during model use.'''


picture_size = 224  # standard resize in vision community for picture cropping/extraction.


# Reads in image and takes center 224x224 slice as input.
def preprocess_image(image):
    if type(image) in (str, unicode):
        image = plt.imread(image)
    crop_edge_h = (image.shape[0] - picture_size) / float(image.shape[0]) / 2
    crop_edge_w = (image.shape[1] - picture_size) / float(image.shape[1]) / 2
    ch = int(image.shape[0] * crop_edge_h + 0.5)
    cw = int(image.shape[1] * crop_edge_w + 0.5)
    cropped_image = image[ch:-ch, cw:-cw]
    if len(cropped_image.shape) == 2:
        cropped_image = np.tile(cropped_image[:, :, np.newaxis], (1, 1, 3))
    return cropped_image


def main():

    # Load parameters from command line.
    indir = FLAGS_indir
    ispy_processed_pickle = FLAGS_ispy_processed_pickle
    ispy_raw = FLAGS_ispy_raw
    outdir = FLAGS_outdir
    extract_only = FLAGS_extract_only
    single_oidx = FLAGS_oidx
    surf_dir = FLAGS_surf_dir

    # Parameters we expect.
    trials = range(1, 6)
    object_ids = range(0, 32)
    if single_oidx is not None:
        object_ids = [single_oidx]
    behaviors = ["drop", "grasp", "hold", "lift", "look", "lower", "press", "push"]
    longest_times = {}  # The longest times per behavior and modality.

    # Audio directory contains a .txt file of numerical features and a .wav file.
    # Haptic directory contains a single .csv of numerical features.
    # Visual directory contains series of images that hopefully number in tandem with frequency of other features.
    # For vision, for now, just storing the filepaths. Can run preprocess_image on those before input to model,
    # otherwise it just takes up too much space on disk.

    # Removes object 0 from ordering_data, which according to Max was a test object.
    # Re-indexes objects 1-32 as objects 0-31 (32 total).

    if extract_only is not None:
        print "loading longest_times structure from file..."
        with open(os.path.join(outdir, "longest_times.pickle"), 'rb') as f:
            longest_times = pickle.load(f)
        print "... done"
    else:
        for b in behaviors:
            longest_times[b] = {}

    # Other I, Spy features.
    print "reading in I, Spy data"
    with open(ispy_processed_pickle, 'rb') as f:
        ispy_features = pickle.load(f)
    print "... done"

    print "extracting features for each object..."
    for oidx in object_ids:
        print str(oidx + 1) + "/" + str(len(object_ids))

        if extract_only is not None:
            print "... loading already-calculated features from file..."
            with open(os.path.join(outdir, str(oidx)+".pickle"), 'rb') as f:
                behavior_obs = pickle.load(f)
            print "... done"
        else:
            behavior_obs = {}

        # Audio.
        if extract_only is None or extract_only == "audio":
            print "... extracting audio data..."
            for b in behaviors:
                obs = []
                longest_times[b]["audio"] = 0
                for tidx in trials:
                    r = os.path.join(indir, "t"+str(tidx), "obj_"+str(oidx+1), "trial_1", b, "audio_data")
                    for root, dirs, files in os.walk(r):
                        for fn in files:
                            if fn.split('.')[-1] == "txt":
                                with open(os.path.join(r, fn), 'r') as f:
                                    d = []
                                    for line in f.readlines():
                                        d.append([float(n) for n in line.split(',')])
                                    d = np.asarray(d)
                                    if len(d) > longest_times[b]["audio"]:
                                        longest_times[b]["audio"] = len(d)
                                    obs.append(d)
                                break
                if b not in behavior_obs:
                    behavior_obs[b] = {"audio": obs}
                else:
                    behavior_obs[b]["audio"] = obs
            print "...... done"

        # Haptic.
        if extract_only is None or extract_only == "haptic":
            print "... extracting haptic data..."
            for b in behaviors:
                obs = []
                longest_times[b]["haptic"] = 0
                for tidx in trials:
                    r = os.path.join(indir, "t"+str(tidx), "obj_"+str(oidx+1), "trial_1", b, "haptic_data")
                    for root, dirs, files in os.walk(r):
                        for fn in files:
                            if fn.split('.')[-1] == "csv":
                                with open(os.path.join(r, fn), 'r') as f:
                                    d = []
                                    for line in f.readlines()[1:]:
                                        d.append([float(n) for n in line.split(',')[:-1]])  # Drop 'timestamp'
                                    d = np.asarray(d)
                                    if len(d) > longest_times[b]["haptic"]:
                                        longest_times[b]["haptic"] = len(d)
                                    obs.append(d)
                                break
                behavior_obs[b]["haptic"] = obs
            print "...... done"

        # Visual.
        print "... extracting vision data..."
        b = "look"
        obs2D = []
        obs3D = []
        obsVGG = []
        if extract_only is None or extract_only == "vision3D":
            longest_times[b]["vision3D"] = 0
        if extract_only is None or extract_only == "visionVGG":
            longest_times[b]["visionVGG"] = 1  # No time dimension in VGG; linear feature rep per object.
        for tidx in trials:

            r = os.path.join(indir, "t"+str(tidx), "obj_"+str(oidx+1), "trial_1", b, "vision_data")
            for root, dirs, files in os.walk(r):
                pic_data = {}  # indexed by pidx
                for fn in files:
                    if fn.split('.')[-1] == "jpg":
                        pidx = int(fn.split('_')[0][4:])
                        pic_data[pidx] = os.path.join(r, fn)
            d = [pic_data[pidx] for pidx in range(min(pic_data.keys()), max(pic_data.keys()))
                 if pidx in pic_data.keys()]
            if len(d) > longest_times[b]["vision3D"]:
                longest_times[b]["vision3D"] = len(d)
            obs3D.append(d)

            if extract_only is None or extract_only == "vision2D":
                pp = [preprocess_image(imgf) for imgf in d]
                a = [[np.mean([pp[idx][hidx][widx][cidx] for idx in range(len(pp))]) / 255.0
                      for cidx in range(0, 3)]
                     for widx in range(224) for hidx in range(224)]
                obs2D.append(a)

        if extract_only is None or extract_only == "visionVGG":
            with open(os.path.join(ispy_raw, "obj"+str(oidx+1), "look", "fc7", "features.csv"), 'r') as f:
                trial_data = f.read().strip().split('\n')
                for td in trial_data:
                    obsVGG.append([float(feat) for feat in td.split(',')[1:]])

        if extract_only is None or extract_only == "vision3D":
            behavior_obs[b]["vision3D"] = obs3D
        if extract_only is None or extract_only == "vision2D":
            behavior_obs[b]["vision2D"] = obs2D
        if extract_only is None or extract_only == "visionVGG":
            behavior_obs[b]["visionVGG"] = obsVGG
        print "...... done"

        # I, Spy features.
        if extract_only is None or extract_only == "ispy":
            print "... getting and storing I, Spy features"
            # skips context 'grasp'-'finger'
            for b in behaviors:
                if b != "look":
                    for mmm, mis in [["audio_ispy", "audio"], ["haptic_ispy", "haptics"]]:
                        behavior_obs[b][mmm] = ispy_features[oidx][b][mis]
                else:
                    for m in ["color", "fpfh"]:
                        behavior_obs[b][m] = ispy_features[oidx][b][m]
            print "...... done"

        # SURF features.
        if extract_only is None or extract_only == "surf":
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

        # Write out.
        print "... writing data to file..."
        with open(os.path.join(outdir, str(oidx)+".pickle"), 'wb') as f:
            pickle.dump(behavior_obs, f)
        print "...... done"
    print "... done"

    # Write out longest times.
    print "writing longest time map to file..."
    with open(os.path.join(outdir, "longest_times.pickle"), 'wb') as f:
        pickle.dump(longest_times, f)
    print "... done"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True,
                        help="directory trial subdirectories live")
    parser.add_argument('--ispy_processed_pickle', type=str, required=True,
                        help="processed ispy features")
    parser.add_argument('--ispy_raw', type=str, required=True,
                        help="directory ispy object data lives in")
    parser.add_argument('--outdir', type=str, required=True,
                        help="outdir for corpus")
    parser.add_argument('--extract_only', type=str, required=False,
                        help="modality to extract exclusively")
    parser.add_argument('--oidx', type=int, required=False,
                        help="single oidx to calculate for")
    parser.add_argument('--surf_dir', type=str, required=False,
                        help="directory in which to locate SURF features")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
