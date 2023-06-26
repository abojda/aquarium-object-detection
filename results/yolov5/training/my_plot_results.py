#!/usr/bin/env python3

from functools import partial
from math import ceil, floor

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def get_fields_config(name):
    x_field = 'epoch'
    linewidth = 2

    if name == 'train_val_loss':
        y_fields = [
            'train/box_loss',
            'train/obj_loss',
            'train/cls_loss',
            'val/box_loss',
            'val/obj_loss',
            'val/cls_loss',
        ]
        ncols = 3

    elif name == 'mAP':
        y_fields = [
            'metrics/mAP_0.5',
            'metrics/mAP_0.5:0.95',
        ]
        ncols = 2

    elif name == 'YOLOv5':
        y_fields = [
            'train/box_loss',
            'train/obj_loss',
            'train/cls_loss',
            'metrics/precision',
            'metrics/recall',
            'val/box_loss',
            'val/obj_loss',
            'val/cls_loss',
            'metrics/mAP_0.5',
            'metrics/mAP_0.5:0.95',
        ]
        ncols = 5

    elif name == 'lr_mAP':
        y_fields = [
            'x/lr0',
            'metrics/mAP_0.5',
            'metrics/mAP_0.5:0.95',
        ]
        ncols = 3

    elif name == 'per_class_mAP':
        y_fields = [
            'metrics/fish_mAP_50',
            'metrics/jellyfish_mAP_50',
            'metrics/penguin_mAP_50',
            'metrics/puffin_mAP_50',
            'metrics/shark_mAP_50',
            'metrics/starfish_mAP_50',
            'metrics/stingray_mAP_50',
            'metrics/mAP_0.5',
            'metrics/mAP_0.5:0.95',
        ]
        ncols = 3

    else:
        ValueError(name)

    return x_field, y_fields, ncols, linewidth


def prepare_plot(nrows, ncols, figsize):
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)

    # Handle 1-row grid
    if nrows == 1:
        ax = ax[np.newaxis, :]

    return fig, ax


def plot(config_name, csv_files, labels, figsize):
    x_field, y_fields, ncols, linewidth = get_fields_config(config_name)
    nrows = ceil(len(y_fields)/ncols)

    fig, ax = prepare_plot(nrows, ncols, figsize)
    pd_frames = [pd.read_csv(f, skipinitialspace=True) for f in csv_files]

    for i, y_field in enumerate(y_fields):
        col = i % ncols
        row = floor(i / ncols)

        ax[row, col].set_title(y_field)

        for frame in pd_frames:
            ax[row, col].plot(frame[x_field],
                              frame[y_field],
                              linewidth=linewidth)

    fig.tight_layout()
    plt.legend(labels)


def available_fields(csv_file):
    df = pd.read_csv(csv_file, skipinitialspace=True)
    return df.columns


if __name__ == '__main__':
    # Plot settings
    sns.set_style('darkgrid')
    plt.rcParams.update({'font.size': 18})

    csv_files = [
        'results_s_default_300/results.csv',
        'results_s_obj0.1_300/results.csv',

        'results_s_default_400/results.csv',
        'results_s_obj0.1_400/results.csv',
    ]

    labels = [f.split('/')[0] for f in csv_files]

    # print(f'Available fields: {available_fields(csv_files[0])}')

    pplot = partial(plot, csv_files=csv_files, labels=labels, figsize=(12, 6))

    pplot('train_val_loss')
    # pplot('mAP')
    # pplot('YOLOv5')
    pplot('lr_mAP')
    # pplot('per_class_mAP')

    plt.show()
