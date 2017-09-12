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
    outdir = FLAGS_outdir
    m = FLAGS_modality

    # Parameters we expect.
    object_ids = range(0, 32)

    # Calculate maximums.
    maximums = None
    minimums = None
    print "calculating minimum and maximum vectors across modality features..."
    for oidx in object_ids:
        print "... from object " + str(oidx)
        with open(os.path.join(indir, str(oidx)+".pickle"), 'rb') as f:
            d = pickle.load(f)
        for b in d:
            if m in d[b]:
                for t in d[b][m]:
                    if type(t[0]) is float:
                        t = [[t[idx] for idx in range(len(t))]]  # wrap as single timestep if binned.
                    if maximums is None:
                        maximums = t[0][:]
                        minimums = t[0][:]
                    for timestep in range(len(t)):
                        if len(t[timestep]) != len(maximums):
                            print ("WARNING: found row with " + str(len(t[timestep])) + " features instead of " +
                                   "established " + str(len(maximums)))
                            continue
                        maximums = [max(maximums[idx], t[timestep][idx]) for idx in range(len(maximums))]
                        minimums = [min(minimums[idx], t[timestep][idx]) for idx in range(len(minimums))]
        print "... max: " + str(maximums)
        print "... min: " + str(minimums)
    print "... done"

    # Normalize and write back files.
    print "normalizing to 0-1 and writing out fresh features..."
    for oidx in object_ids:
        print "... normalizing object " + str(oidx)
        with open(os.path.join(indir, str(oidx)+".pickle"), 'rb') as f:
            d = pickle.load(f)
        for b in d:
            if m in d[b]:
                for tidx in range(len(d[b][m])):
                    timesteps_to_remove = []
                    if type(d[b][m][tidx][0]) is float:
                        d[b][m][tidx] = [[d[b][m][tidx][idx] for idx in range(len(d[b][m][tidx]))]]
                    for timestep in range(len(d[b][m][tidx])):
                        if len(d[b][m][tidx][timestep]) != len(maximums):
                            print ("WARNING: removing row with " + str(len(d[b][m][tidx][timestep])) +
                                   " features instead of established " + str(len(maximums)))
                            timesteps_to_remove.append(timestep)
                            continue
                        d[b][m][tidx][timestep] = [(d[b][m][tidx][timestep][idx] - minimums[idx]) /
                                                   (maximums[idx] - minimums[idx])
                                                   for idx in range(len(maximums))
                                                   if maximums[idx] - minimums[idx] > 0]
                    d[b][m][tidx] = [d[b][m][tidx][timestep] for timestep in range(len(d[b][m][tidx]))
                                     if timestep not in timesteps_to_remove]
        print "... writing features back to file"
        with open(os.path.join(outdir, str(oidx)+".pickle"), 'rb') as f:
            d_to_write = pickle.load(f)
            for b in d_to_write:
                if m in d[b]:
                    d_to_write[b][m] = d[b][m]
        with open(os.path.join(outdir, str(oidx)+".pickle"), 'wb') as f:
            pickle.dump(d_to_write, f)
    print "... done"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True,
                        help="un-normalized corpus directory")
    parser.add_argument('--outdir', type=str, required=True,
                        help="normalized corpus directory")
    parser.add_argument('--modality', type=str, required=True,
                        help="which modality to normalize")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
