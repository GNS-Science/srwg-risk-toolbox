from .base import *

def call_sa_parameters(site, rp, sc, sa_table):
    line = [line for line in sa_table if
            (line['Location'] == site) & (line['Site Soil Class'] == sc) & (line['APoE (1/n)'] == rp)][0]

    return line['PGA'], line['Sas'], line['Tc']


def retrieve_ts_design_im(site, rp, sc, period, sa_table):
    pga, sas, tc = call_sa_parameters(site, rp, sc, sa_table)

    return uhs_value(period, pga, sas, tc)



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


def infer_cmr_fragilities(cmr_median, beta_cmr, beta_rtr, p=0.9):
    ''' use cmr distribution's median and dispersion, beta_rtr, and percentile to calculate
        other parameters of the fragility distribution and return as dictionary

    :param cmr_median: median value of the distribution of cmr values (fragility medians)
    :param beta_comp: lognormal standard deviation of the distribution of the cmrs
    :param beta_rtr: record-to-record uncertainty of each fragility
    :param p: percentile of interest for the pth percentile of the risk

    :return: dictionary of the relevant parameters for the fragility distribution
    '''

    beta_tot = np.sqrt(beta_cmr ** 2 + beta_rtr ** 2).round(2)
    cmr_p = stats.lognorm(beta_cmr, scale=cmr_median).ppf(1 - p).round(1)

    return {"cmr_median": cmr_median, "cmr_p": cmr_p, "beta_tot": beta_tot, "beta_rtr": beta_rtr, "beta_cmr": beta_cmr}


def calc_fatality_risk_stats(fragility_parameters, buffer, dil2, hcurve, imtl):
    ''' calculate statistics of the risk distribution, based on design inputs and the hazard curve

    :param fragility_parameters: dictionary of the relevant parameters for the fragility distribution
    :param buffer: multiple of the dil2 to be used as the design value for a non-ductile building
    :param dil2: baseline design value for life-safety of a ductile building
    :param hcurve: hazard curve annual probabilities of exceedance
    :param imtl: hazard curve intensity measures

    :return: list of risk values - mean, median, and percentile of interest (based on fragility_parameters)
    '''
    p_fatality_given_collapse = 0.1

    cmr_median = fragility_parameters['cmr_median']
    cmr_p = fragility_parameters['cmr_p']
    beta_tot = fragility_parameters['beta_tot']
    beta_rtr = fragility_parameters['beta_rtr']

    median = cmr_median * dil2 * buffer
    beta = beta_tot
    collapse_risk, _ = risk_convolution(hcurve, imtl, median, beta)
    risk_mean = collapse_risk * p_fatality_given_collapse

    median = cmr_median * dil2 * buffer
    beta = beta_rtr
    collapse_risk, _ = risk_convolution(hcurve, imtl, median, beta)
    risk_median = collapse_risk * p_fatality_given_collapse

    median = cmr_p * dil2 * buffer
    beta = beta_rtr
    collapse_risk, _ = risk_convolution(hcurve, imtl, median, beta)
    risk_p = collapse_risk * p_fatality_given_collapse

    return risk_mean, risk_median, risk_p
