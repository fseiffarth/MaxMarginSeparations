'''
Created on 19.11.2018

@author: florian
'''

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator


def save_3d_data_colorbar(path, name, data, x_label_list, y_label_list, x_label_name_list=None, y_label_name_list=None,
                          min_val=0, max_val=100, heading="Test", x_label_name="xLabel", y_label_name="yLabel",
                          colormap="Reds", colorbar_label=""):
    plot_3d_data_colorbar(data, x_label_list, y_label_list, x_label_name_list, y_label_name_list, min_val, max_val,
                          heading, x_label_name, y_label_name, colormap, colorbar_label, path)


def plot_3d_data_colorbar(data, y_label_list, x_label_list, y_label_name_list=None, x_label_name_list=None, min_val=0,
                          max_val=100, heading="Test", x_label_name="xLabel", y_label_name="yLabel", colormap="Reds",
                          colorbar_label="", tikz_save=None, name="Test"):
    '''
    Plots or saves 3d data with colorbar

    :param data: 2d numpy array of data values
    :param x_label_list: list of x-label values for the plot ticks
    :param y_label_list: list of E_labels-label values for the plot ticks
    :param x_label_name_list: list of x-label names for the ticks
    :param y_label_name_list: list of E_labels-label names for the ticks
    :param min_val: minimum value for the colorbar
    :param max_val: maximum value for the colorbar
    :param heading: heading of the plot
    :param x_label_name: x-label name of the plot
    :param y_label_name: E_labels-label name of the plot
    :param colormap: colormap of the plot
    :param colorbar_label: label of the colorbar of the plot
    :param tikz_save: if false then data is plotted, else give path where tikz file is stored
    :param name: name of the saved tikz file
    :return None: 
    '''

    if y_label_name_list is None:
        y_label_name_list = y_label_list

    if y_label_name_list is None:
        y_label_name_list = y_label_list

    if x_label_list == "class":
        x_label_list = [x for x in range(0, len(x_label_name_list))]

    if y_label_list == "class":
        y_label_list = [x for x in range(0, len(y_label_name_list))]

    print(x_label_name_list, x_label_list)

    levels = MaxNLocator(nbins=200).tick_values(min_val, max_val)

    cmap = plt.get_cmap(colormap)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    fig, ax0 = plt.subplots()

    x_label_list = np.append(x_label_list, int(x_label_list[len(x_label_list) - 1]) + 1)
    y_label_list = np.append(y_label_list, int(y_label_list[len(y_label_list) - 1]) + 1)

    x_labels = np.array([x for x in range(len(x_label_list))])
    y_labels = np.array([y for y in range(len(y_label_list))])

    im = ax0.pcolormesh(x_labels - 1. / 2., y_labels - 1. / 2., data, cmap=cmap, norm=norm)

    cbar = fig.colorbar(im, ax=ax0)
    cbar.set_label(colorbar_label, rotation=270)

    ax0.set_title(heading)

    fig.tight_layout()
    plt.xlabel(x_label_name)
    plt.ylabel(y_label_name)

    plt.xticks(x_labels[:-1], x_label_name_list)
    plt.yticks(y_labels[:-1], y_label_name_list)

    if tikz_save is not None:
        tikz_save(tikz_save + name + ".tex", wrap=False)
    else:
        plt.show()
