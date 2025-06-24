"""
This module extracts the data and metadata in the hdf5 containing the NSHM data.
"""
import h5py
import ast
import pandas as pd

from typing import TYPE_CHECKING, Tuple, List


def extract_spectra(data_file, intensity_type):
    """Extract the uniform hazard spectra from the hdf5

    Args:
        data_file: name of hazard hdf5 file
        intensity_type: 'acc' for acceleration, 'disp' for displacement

    Returns:
        spectra: spectra
        imtls: keys: intensity measures e.g., SA(1.0), values: list of intensity levels

    """
    with h5py.File(data_file, "r") as hf:
        imtls = ast.literal_eval(hf["metadata"].attrs[f"{intensity_type}_imtls"])
        spectra = hf["hazard_design"][intensity_type]["stats_im_hazard"][:]

    return spectra, imtls


def extract_single_uhs_value(data_file, intensity_type, site, vs30, rp, imt, metric):
    """Extract a single uniform hazard spectra from the hdf5
    """

    spectra, imtls = extract_spectra(data_file, intensity_type)

    site_list = list(extract_sites(data_file).index)
    vs30_list = extract_vs30s(data_file)
    imt_list = list(imtls.keys())
    _, hazard_rp_list = extract_APoEs(data_file)
    quantiles = extract_quantiles(data_file)

    i_site = site_list.index(site)
    i_vs30 = vs30_list.index(vs30)
    i_imt = imt_list.index(imt)
    i_rp = hazard_rp_list.index(rp)

    if metric=='mean':
        i_metric = 0
    else:
        i_metric = quantiles.index(metric) + 1

    return spectra[i_vs30,i_site,i_imt,i_rp,i_metric]

def extract_vs30s(data_file):
    """Extract the vs30 values from the hdf5

    Args:
        data_file: name of hazard hdf5 file

    Returns:
        vs30_list: list of vs30s included in hdf5

    """
    with h5py.File(data_file, "r") as hf:
        vs30_list = list(hf["metadata"].attrs["vs30s"])

    return vs30_list


def extract_quantiles(data_file):
    """Extract hazard quantiles from the hdf5

    Args:
        data_file: name of hazard hdf5 file

    Returns:
        quantiles: list of quantiles

    """
    with h5py.File(data_file, "r") as hf:
        quantiles = list(hf["metadata"].attrs["quantiles"])

    return quantiles


def extract_sites(data_file):
    """Extract sites from the hdf5

    Args:
        data_file: name of hazard hdf5 file

    Returns:
        sites: dataframe of sites with lat/lons

    """
    with h5py.File(data_file, "r") as hf:
        sites = pd.DataFrame(ast.literal_eval(hf["metadata"].attrs["sites"]))

    return sites


def extract_APoEs(data_file):
    """Extract uniform hazard spectra annual probabilities of exceedance from the hdf5

    Args:
        data_file: name of hazard hdf5 file

    Returns:
        APoEs: list of APoE strings
        hazard_rp_list: list of return periods

    """
    with h5py.File(data_file, "r") as hf:
        hazard_rp_list = list(hf["hazard_design"].attrs["hazard_rps"])
    APoEs = [f"APoE: 1/{hazard_rp}" for hazard_rp in hazard_rp_list]

    return APoEs, hazard_rp_list
