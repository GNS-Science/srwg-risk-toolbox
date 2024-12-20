import pandas as pd
import numpy as np
import h5py
import ast

from srwg_risk_toolbox.base import period_from_imt, choose_representative_vs30, uhs_value
from srwg_risk_toolbox.liquefaction.liquefaction_model import calc_pga_7pt5, SiteHazard, building_settlement_from_sources


from pathlib import Path
from scipy.interpolate import interp1d

data_folder = Path(r'C:\Users\ahul697\OneDrive - The University of Auckland\Desktop\Research\GitHub_Repos\GNS\srwg-risk-toolbox\data')

filename = Path(data_folder,'named_locations_combo.json')
TS_TABLE = pd.read_json(filename,orient='table',precise_float=True)


hcurves_file = 'hcurves_12-locations_all-NSHM-parameters.hdf5'

with h5py.File(Path(data_folder, hcurves_file), 'r') as hf:
    hf_vs30_list = list(hf['metadata'].attrs['vs30s'])
    sites = pd.DataFrame(ast.literal_eval(hf['metadata'].attrs['sites']))
    hf_site_list = list(sites.index)
    imtls = ast.literal_eval(hf['metadata'].attrs['acc_imtls'])
    hf_imt_list = list(imtls.keys())
    hf_period_list = [period_from_imt(imt) for imt in hf_imt_list]
    hf_quantiles_list = list(hf['metadata'].attrs['quantiles'])

    hcurves = hf['hcurves']['hcurves_stats'][:]

    imt = 'PGA'
    hf_i_imt = hf_imt_list.index(imt)
    imtl = imtls[imt]


# def query_pga_7pt5(site, sc):
#     site_idx = TS_TABLE['Location'] == site
#     sc_idx = TS_TABLE['Site Class'] == sc
#     pga = TS_TABLE[site_idx & sc_idx][['APoE (1/n)', 'PGA', 'M']].set_index('APoE (1/n)')
#     rp_list = pga.index
#
#     for rp in rp_list:
#         pga.loc[rp, 'PGA_7pt5'] = calc_pga_7pt5(pga.loc[rp, 'PGA'], pga.loc[rp, 'M'])
#
#     return pga


def query_ts_hazard(site, sc):
    site_idx = TS_TABLE['Location'] == site
    sc_idx = TS_TABLE['Site Class'] == sc
    site_hazard = TS_TABLE[site_idx & sc_idx][['APoE (1/n)', 'PGA', 'M', 'Sas', 'Tc']].set_index('APoE (1/n)')
    rp_list = site_hazard.index

    period = 1
    td = 3  # placeholder, as Td doesn't affect Sa1
    for rp in rp_list:
        pga = site_hazard.loc[rp, 'PGA']
        m = site_hazard.loc[rp, 'M']
        site_hazard.loc[rp, 'PGA_7pt5'] = calc_pga_7pt5(pga, m)

        sas = site_hazard.loc[rp, 'Sas']
        tc = site_hazard.loc[rp, 'Tc']
        site_hazard.loc[rp, 'Sa1'] = uhs_value(period,pga,sas,tc,td)

    return site_hazard


def query_settlement_hazard(site, sc, liquefaction_parameters):

    site_apoes_hazard = query_ts_hazard(site, sc)

    for rp in site_apoes_hazard.index:
        pga, m_w, sa1, pga_7pt5 = site_apoes_hazard.loc[rp, ['PGA', 'M', 'Sa1', 'PGA_7pt5']]
        liquefaction_parameters['site_hazard'] = SiteHazard(pga, m_w, sa1, pga_7pt5)

        sb, _ = building_settlement_from_sources(liquefaction_parameters)
        site_apoes_hazard.loc[rp, ['SB']] = sb

    return site_apoes_hazard


def extrapolate_hcurve(return_periods, ims, imtl, kind='linear'):
    x = np.log(ims)
    y = np.log(1 / np.array(return_periods))
    f = interp1d(x, y, bounds_error=False, kind=kind, fill_value='extrapolate')

    hcurve = np.exp(f(np.log(imtl)))

    return hcurve


def calculate_hcurve(imt,hazard_table,imtl=None):

    if imtl is None:
        max_x = 2 * hazard_table[imt].max()
        imtl = np.linspace(0,max_x,int(1e3))

    return_periods = hazard_table.index.to_numpy()
    ims = hazard_table[imt].to_numpy()

    hcurve = extrapolate_hcurve(return_periods, ims, imtl)

    return hcurve, imtl


def retrieve_pga_hcurve(site, sc, metric='mean'):

    if metric!='mean':
        hf_i_metric = 1 + hf_quantiles_list.index(metric)
    else:
        hf_i_metric = 0

    vs30 = choose_representative_vs30(sc)

    hf_i_site = hf_site_list.index(site)
    hf_i_vs30 = hf_vs30_list.index(vs30)
    nshm_hcurve = hcurves[hf_i_vs30, hf_i_site, hf_i_imt, :, hf_i_metric]

    return nshm_hcurve, imtl


