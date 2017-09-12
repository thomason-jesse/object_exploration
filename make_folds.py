#!/usr/bin/env python

import argparse
import pickle
import random


'''Extracts data from flat files and stores it in more coherent pickles for haptic and audio features.
Does not perform image feature extraction, but instead writes a list of image paths to be processed
on-the-fly during model use.'''


def main():

    # Load parameters from command line.
    nb_objects = FLAGS_nb_objects
    outfile = FLAGS_outfile
    props = [float(p) for p in FLAGS_props.split(',')]
    assert nb_objects >= len(props)

    # Shuffle object IDs and divide.
    oidxs = range(nb_objects)
    random.shuffle(oidxs)
    last = 0
    last_idx = 0
    fold_obs = []
    for idx in range(0, len(props)):
        nb_obs_c = props[idx] * nb_objects
        next_idx = int(last + nb_obs_c + 0.5)
        fold_obs.append(oidxs[last_idx:next_idx])
        print "fold " + str(idx) + " got " + str(next_idx-last_idx) + " objects: " + str(fold_obs[-1])
        last += nb_obs_c
        last_idx = next_idx

    # Write fold -> oidxs to file.
    with open(outfile, 'wb') as f:
        pickle.dump(fold_obs, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nb_objects', type=int, required=True,
                        help="number of objects to divide")
    parser.add_argument('--props', type=str, required=True,
                        help="comma separated proportions")
    parser.add_argument('--outfile', type=str, required=True,
                        help="place to write list of object ids per fold")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
