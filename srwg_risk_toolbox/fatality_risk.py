from .base import *

def period_from_imt(imt):
    if imt in ['PGA','PGD']:
        period = 0
    else:
        period = float(re.split('\(|\)',imt)[1])
    return period


def uhs_value(period, PGA, Sas, Tc, Td=3):
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

    return value


def choose_representative_vs30(sc):
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

    return sc_dict[sc]['representative_vs30']


def choose_site_class(vs30, lower_bound=False):
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

    boundaries = np.array([sc['lower_bound'] for sc in sc_dict.values()][:-1])

    # switches which SC is selected at the boundary
    if lower_bound:
        side = 'left'
    else:
        side = 'right'

    sc_idx = np.searchsorted(-boundaries, -vs30, side=side)

    return list(sc_dict.keys())[sc_idx]


def sample_cmrs(n_samples, mean_cmr=6, std_cmr=1.5, min_cmr=3, max_cmr=9):
    sampled_cmrs = stats.norm.rvs(mean_cmr, std_cmr, n_samples)
    while np.any(sampled_cmrs < min_cmr) | np.any(sampled_cmrs > max_cmr):
        n_resamples = sum(sampled_cmrs < min_cmr)
        sampled_cmrs[sampled_cmrs < min_cmr] = stats.norm.rvs(mean_cmr, std_cmr, n_resamples)
        n_resamples = sum(sampled_cmrs > max_cmr)
        sampled_cmrs[sampled_cmrs > max_cmr] = stats.norm.rvs(mean_cmr, std_cmr, n_resamples)

    return sampled_cmrs


def sample_betas(n_samples, min_beta=0.35, max_beta=0.45):
    return stats.uniform.rvs(min_beta, max_beta - min_beta, n_samples)


def risk_convolution(hcurve, imtl, median, beta):
    '''
    calculates the total annual risk and the underlying disaggregation curve

    :param hcurve: hazard curve
    :param imtl:   intensity measure levels
    :param median: median of the fragility function
    :param beta:   log std for the fragility function

    :return: the total risk and the disagg curve
    '''

    pdf_limitstate_im = stats.lognorm(beta, scale=median).pdf(imtl)
    disaggregation = pdf_limitstate_im * hcurve
    risk = np.trapz(disaggregation, x=imtl)

    return risk, disaggregation
