import numpy as np
import pandas as pd
from scipy import stats
import re
import h5py
import ast
import json
import matplotlib.pyplot as plt

def set_plot_formatting():
    # set up plot formatting
    SMALL_SIZE = 15
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 25

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('legend', title_fontsize=SMALL_SIZE)  # legend title fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


from rich.tree import Tree
from rich import print as rprint


def create_dictionary_tree(dct: dict, name='Dictionary', description=False) -> None:
    """Recursively build a Tree with dictionary contents."""
    tree = Tree(name)
    for key in dct.keys():
        if isinstance(dct[key], dict):
            branch = tree.add(f'{key}')
            walk_dictionary(dct[key], branch, description)
        else:
            if description:
                label = f' ({type(dct[key])})'
                print('x')
            else:
                label = ''
            branch = tree.add(f'{key}{label}')

    rprint(tree)


def walk_dictionary(dct: dict, tree: Tree, description=False) -> None:
    """Recursively build a Tree with dictionary contents."""
    for key in dct.keys():
        if isinstance(dct[key], dict):
            branch = tree.add(f'{key}')
            walk_dictionary(dct[key], branch, description)
        else:
            if description:
                label = f' ({type(dct[key])})'
            else:
                label = ''
            branch = tree.add(f'{key}{label}')
