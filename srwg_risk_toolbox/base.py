import numpy as np
import pandas as pd
from scipy import stats
import re
import h5py
import ast
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import minimize
import os
from pathlib import Path
from rich.tree import Tree
from rich import print as rprint

g = 9.80665 # gravity in m/s^2

def acc_to_disp(acc,t):
    return (acc * g) * (t/(2*np.pi))**2

def acc_to_vel(acc,t):
    return (acc * g) * (t/(2*np.pi))


def disp_to_acc(disp,t):
    return disp / (t/(2*np.pi))**2 / g

def vel_to_acc(disp,t):
    return disp / (t/(2*np.pi)) / g

def period_from_imt(imt):

    if imt in ['PGA','PGD']:
        period = 0
    else:
        period = float(re.split(r'\(|\)',imt)[1])

    return period


def imt_from_period(period):
    if period == 0:
        imt = 'PGA'
    else:
        imt = f'SA({period})'
    return imt


def prob_in_n_years(risk,n_years=50):

    return 1 - np.exp(-risk * n_years)


def calculate_apoe_intensity(hazard_rp, hcurve, imtl):
    '''

    :param hazard_rp: int   return period, inverse of the annual probability of exceedance (apoe)
    :param hcurve: list or np.array     list of apoes corresponding to the intensities
    :param imtl:   list or np.array     list of intensities

    :return: float  apoe intensity, linearly interpolated in log space
    '''

    return np.exp(np.interp(np.log(1 / hazard_rp), np.log(np.flip(hcurve)), np.log(np.flip(imtl))))


sc_dict = {'I': {'representative_vs30': 750,
                     'label': 'Site Soil Class I',
                     'lower_bound': 750,
                     'upper_bound': np.nan},
               'II': {'representative_vs30': 525,
                      'label': 'Site Soil Class II',
                      'lower_bound': 450,
                      'upper_bound': 750},
               'III': {'representative_vs30': 375,
                       'label': 'Site Soil Class III',
                       'lower_bound': 300,
                       'upper_bound': 450},
               'IV': {'representative_vs30': 275,
                      'label': 'Site Soil Class IV',
                      'lower_bound': 250,
                      'upper_bound': 300},
               'V': {'representative_vs30': 225,
                     'label': 'Site Soil Class V',
                     'lower_bound': 200,
                     'upper_bound': 250},
               'VI': {'representative_vs30': 175,
                      'label': 'Site Soil Class VI',
                      'lower_bound': 150,
                      'upper_bound': 200},
               'VII': {'representative_vs30': 150,
                       'label': 'Site Soil Class VII',
                       'lower_bound': np.nan,
                       'upper_bound': 150}}


def choose_representative_vs30(sc):

    return sc_dict[sc]['representative_vs30']


def choose_site_class(vs30, lower_bound=False):

    boundaries = np.array([sc['lower_bound'] for sc in sc_dict.values()][:-1])

    # switches which SC is selected at the boundary
    if lower_bound:
        side = 'left'
    else:
        side = 'right'

    sc_idx = np.searchsorted(-boundaries, -vs30, side=side)

    return list(sc_dict.keys())[sc_idx]


def uhs_value(period, PGA, Sas, Tc, Td, decimal_places=3):
    if period == 0:
        value = PGA
    elif period < 0.1:
        value = PGA + (Sas - PGA) * (period / 0.1)
    elif period < Tc:
        value = Sas
    elif period < Td:
        value = Sas * Tc / period
    else:
        value = Sas * Tc / period * (Td / period) ** 0.5

    return np.round(value, decimal_places)


def convert_acc_imtls_to_disp(acc_imtls):
    '''
    converts the acceleration intensity measure types and levels to spectral displacements
    '''
    disp_imtls = {}
    for acc_imt in acc_imtls.keys():
        period = period_from_imt(acc_imt)
        disp_imt = acc_imt.replace('A', 'D')

        disp_imtls[disp_imt] = acc_to_disp(np.array(acc_imtls[acc_imt]), period).tolist()

    return disp_imtls


def interpolate_hcurves(hcurves, imtls, n_new_imtls=300):

    imts = list(imtls.keys())
    imtl_list = imtls[imts[0]]
    min_imtl = imtl_list[0]
    max_imtl = imtl_list[-1]

    new_imtl_list = np.logspace(np.log10(min_imtl), np.log10(max_imtl), n_new_imtls)
    new_imtls = {}
    for imt in imts:
        new_imtls[imt] = list(new_imtl_list)

    n_vs30, n_sites, n_imts, n_imtls, n_stats = hcurves.shape
    new_hcurves = np.zeros([n_vs30, n_sites, n_imts, n_new_imtls, n_stats])

    for i_vs30 in range(n_vs30):
        for i_site in range(n_sites):
            for i_imt in range(n_imts):
                for i_stat in range(n_stats):
                    hcurve = hcurves[i_vs30, i_site, i_imt, :, i_stat]
                    new_hcurves[i_vs30, i_site, i_imt, :, i_stat] = np.exp(
                        np.interp(np.log(new_imtl_list), np.log(imtl_list), np.log(hcurve)))

    return new_hcurves, new_imtls


def find_fragility_median(im_value, beta, ls_prob):
    return minimize(ls_fragility_median_error, im_value, args=(im_value, beta, ls_prob), method='Nelder-Mead').x[0]


def ls_fragility_median_error(median, im, beta, target_prob):
    return np.abs(target_prob - stats.lognorm(beta, scale=median).cdf(im))[0]


def find_fragility_beta(median, im_value, ls_prob):
    beta_0 = 0.5
    return minimize(ls_fragility_beta_error, beta_0, args=(median, im_value, ls_prob), method='Nelder-Mead').x[0]


def ls_fragility_beta_error(beta, median, im, target_prob):
    return np.abs(target_prob - stats.lognorm(beta, scale=median).cdf(im))[0]

def risk_convolution_error(median, hcurve, imtl, beta, target_risk):
    '''
    error function for optimization

    :param median: median of the fragility function
    :param hcurve: hazard curve
    :param imtl:   intensity measure levels
    :param beta:   log std for the fragility function
    :param target_risk:  risk value to target

    :return: error from risk target
    '''
    # the derivative of the fragility function, characterized as the pdf instead of the cdf
    pdf_limitstate_im = stats.lognorm(beta, scale=median).pdf(imtl)
    disaggregation = pdf_limitstate_im * hcurve
    risk = np.trapz(disaggregation, x=imtl)

    return np.abs(target_risk - risk)

def find_uniform_risk_intensity(hcurve, imtl, beta, target_risk, design_point):
    '''
    optimization to find the fragility and associated design intensity

    :param hcurve: hazard curve
    :param imtl:   intensity measure levels
    :param beta:   log std for the fragility function
    :param target_risk:   risk value to target
    :param design_point:  design point for selecting the design intensity

    :return: design intensity and median of fragility
    '''

    x0 = 0.5
    median = minimize(risk_convolution_error, x0, args=(hcurve, imtl, beta, target_risk), method='Nelder-Mead').x[0]
    im_r = stats.lognorm(beta, scale=median).ppf(design_point)

    return im_r, median



def set_plot_formatting(SMALL_SIZE=15,MEDIUM_SIZE=18,BIGGER_SIZE=25):
    # set up plot formatting

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('legend', title_fontsize=SMALL_SIZE)  # legend title fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


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
