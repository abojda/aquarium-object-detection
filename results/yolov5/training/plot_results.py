#!/usr/bin/env python3

import argparse
from yolov5.utils.plots import plot_results
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Plot results',
        description='Plot YOLOv5 training results from .csv file')

    parser.add_argument('-d', '--dir', default='.')

    args = parser.parse_args()

    sns.set_style('darkgrid')
    plot_results(file='', dir=args.dir)
    plt.legend(loc='upper right')
