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
