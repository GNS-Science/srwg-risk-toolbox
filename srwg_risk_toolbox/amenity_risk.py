from .base import *

from .fatality_risk import risk_convolution


def cdf_to_pdf(cdf, imtl):
    '''

    :param cdf:  np.array     cumulative distribution function corresponding to the intensities
    :param imtl: np.array     list of intensities

    :return: np.array         derivitative of the cdf, or the probability density function, pdf
    '''

    return np.gradient(cdf, imtl)


def pdf_risk_convolution(pdf, hcurve, imtl):
    '''

    :param pdf:    np.array     probability density function corresponding to the intensities
    :param hcurve: np.array     list of annual probability of exceedance (apoe) corresponding to the intensities
    :param imtl:   np.array     list of intensities

    :return: float      annual risk
             np.array   disaggregation of risk corresponding to the intensities
    '''

    disaggregation = pdf * hcurve
    risk = np.trapz(disaggregation, x=imtl)

    return risk, disaggregation


def calc_slope_and_intercept(pt_1, pt_2):
    '''

    :param pt_1: tuple  (x,y) of first point
    :param pt_2: tuple  (x,y) of second point

    :return: float  slope of line between pt1 and pt2
             float  intercept of line
    '''

    pts = [pt_1, pt_2]

    m = (pts[0][1] - pts[1][1]) / (pts[0][0] - pts[1][0])
    b = (pts[0][0] * pts[1][1] - pts[1][0] * pts[0][1]) / (pts[0][0] - pts[1][0])

    return m, b


def loss_model(imtl, pt_low, pt_high, pt_corner=None, total_loss_triggered=None, plot_cdf=False, plot_pdf=True):
    '''

    :param imtl:     list or np.array     list of intensities
    :param pt_low:   tuple  (x,y) of first point defining the cumulative distribution function, cdf
    :param pt_high:  tuple  (x,y) of last point defining the cumulative distribution function, cdf
    :param pt_corner: None or tuple  (x,y) of bilinear point in the cumulative distribution function, cdf
    :param total_loss_triggered: None or float  0 < t_l_t < 1  loss above which total loss is triggered
    :param plot_cdf:  boolean  True produces a plot of the cdf
    :param plot_pdf:  boolean  True also includes a plot of the proability density function

    :return: np.array   cumulative distribution function corresponding to the intensities
             np.array   probability density frunction
    '''

    imtl = np.array(imtl)

    if pt_corner is not None:
        cdf = np.zeros_like(imtl)

        for pt_1, pt_2 in zip([pt_low, pt_corner], [pt_corner, pt_high]):
            m, b = calc_slope_and_intercept(pt_1, pt_2)
            y = m * imtl + b
            cdf = np.maximum(cdf, y)

    else:
        m, b = calc_slope_and_intercept(pt_low, pt_high)
        cdf = m * imtl + b

    cdf[cdf < 0] = 0
    cdf[cdf > 1] = 1

    if total_loss_triggered is not None:
        cdf[cdf > total_loss_triggered] = 1

    pdf = cdf_to_pdf(cdf, imtl)

    if plot_cdf:
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        ax = [ax]

        i_ax = 0
        color = 'k'
        _ = ax[i_ax].plot(imtl, cdf, color=color, lw=2, label='CDF')
        for pt in [pt_low, pt_high]:
            _ = ax[i_ax].scatter(pt[0], pt[1], color=color)

        if pt_corner is not None:
            for pt in [pt_corner]:
                _ = ax[i_ax].scatter(pt[0], pt[1], color=color)

        _ = ax[i_ax].set_xlim(left=0)
        _ = ax[i_ax].set_ylim([0, 1])
        _ = ax[i_ax].set_ylabel('Expected Loss')
        _ = ax[i_ax].set_xlabel('Intensity Measure, IM')

        if total_loss_triggered is not None:
            xlim = ax[i_ax].get_xlim()
            _ = ax[i_ax].plot(xlim, [total_loss_triggered] * 2, ls=':', color=color)

        if plot_pdf:
            ax = ax + [ax[0].twinx()]
            i_ax = 1

            area_under_pdf = np.trapz(pdf, imtl)

            color = 'lightgray'
            _ = ax[0].fill_between([0], [0], [0], color=color, label='PDF')
            _ = ax[i_ax].fill_between(imtl, 0, pdf, color=color, label=f'{area_under_pdf:.2f}', zorder=-1)

            _ = ax[i_ax].set_ylim(bottom=0)
            _ = ax[i_ax].set_ylabel('Derivative of Expected Loss')

        i_ax = 0
        _ = ax[i_ax].patch.set_visible(False)
        _ = ax[i_ax].set_zorder(ax[i_ax].get_zorder() + 1)
        _ = ax[i_ax].legend()
        _ = plt.show()

    return cdf, pdf



def infer_component_fragilities(comp_median, beta_comp, beta_rtr, p=0.9):
    ''' use component distribution's median and dispersion, beta_rtr, and percentile to calculate
        other parameters of the fragility distribution and return as dictionary

    :param comp_median: median value of the distribution of component fragility medians
    :param beta_comp: lognormal standard deviation of the distribution of the component fragility medians
    :param beta_rtr: record-to-record uncertainty of each fragility
    :param p: percentile of interest for the pth percentile of the risk

    :return: dictionary of the relevant parameters for the fragility distribution
    '''
    beta_tot = np.sqrt(beta_comp ** 2 + beta_rtr ** 2).round(2)
    comp_p = stats.lognorm(beta_comp, scale=comp_median).ppf(1 - p).round(4)

    return {"comp_median": comp_median, "comp_p": comp_p, "beta_tot": beta_tot, "beta_rtr": beta_rtr,
            "beta_comp": beta_comp}


def comp_id_from_desc(comp_name, desc, component_data):
    ''' searches a dictionary of component data, looking for the component associated with a description

    :param comp_name: name of the general class of components (e.g., Partitions)
    :param desc: description to search for
    :param component_data: dictionary of component data

    :return: comp_id: id of the relevant component
    '''

    components = component_data[comp_name]
    comp_id = [comp_id for comp_id, comp_info in components.items() if comp_info['description'] == desc][0]
    return comp_id


def calc_component_damage_risk_stats(fragility_parameters, hcurve, imtl, risk_duration=50):
    ''' calculate statistics of the risk distribution, based on design inputs and the hazard curve

    :param fragility_parameters: dictionary of the relevant parameters for the fragility distribution
    :param hcurve: hazard curve annual probabilities of exceedance
    :param imtl: hazard curve intensity measures -- already adjusted for drift
    :param risk_duration: length of time considered (e.g., risk in 50 years)

    :return: list of risk values - mean, median, and percentile of interest (based on fragility_parameters)
    '''
    comp_median = fragility_parameters['comp_median']
    comp_p = fragility_parameters['comp_p']
    beta_tot = fragility_parameters['beta_tot']
    beta_rtr = fragility_parameters['beta_rtr']

    risk_metrics = []

    for i, (median, beta) in enumerate(zip([comp_median, comp_median, comp_p], [beta_tot, beta_rtr, beta_rtr])):
        risk, _ = risk_convolution(hcurve, imtl, median, beta)
        risk_metrics.append(prob_in_n_years(risk, risk_duration))

    return risk_metrics


def calc_component_damage_risk_stats_alt_pvalues(fragility_parameters, hcurve, imtl, p_values=None, risk_duration=50):
    ''' calculate statistics of the risk distribution, based on design inputs and the hazard curve
    similar to calc_component_damage_risk_stats except that additional p values can be passed in

    :param fragility_parameters: dictionary of the relevant parameters for the fragility distribution
    :param hcurve: hazard curve annual probabilities of exceedance
    :param imtl: hazard curve intensity measures -- already adjusted for drift
    :param p_values: list of p values to calculate risk for
    :param risk_duration: length of time considered (e.g., risk in 50 years)

    :return: list of risk values - mean, median, and percentile of interest (based on fragility_parameters)
    '''

    comp_median = fragility_parameters['comp_median']
    comp_p = fragility_parameters['comp_p']
    beta_tot = fragility_parameters['beta_tot']
    beta_rtr = fragility_parameters['beta_rtr']
    beta_comp = fragility_parameters['beta_comp']

    if p_values is None:
        comp_ps = [comp_p]
    else:
        comp_ps = [stats.lognorm(beta_comp, scale=comp_median).ppf(1 - p).round(4) for p in p_values]
    beta_rtr_ps = [beta_rtr] * len(comp_ps)

    medians = [comp_median] * 2 + comp_ps
    betas = [beta_tot, beta_rtr] + beta_rtr_ps

    risk_metrics = []

    for i, (median, beta) in enumerate(zip(medians, betas)):
        risk, _ = risk_convolution(hcurve, imtl, median, beta)
        risk_metrics.append(prob_in_n_years(risk, risk_duration))

    return risk_metrics


def infer_dmr_fragilities(dmr_median, beta_dmr, beta_rtr, p=0.9):
    ''' use dmr distribution's median and dispersion, beta_rtr, and percentile to calculate
        other parameters of the fragility distribution and return as dictionary

    :param cmr_median: median value of the distribution of dmr values (fragility medians)
    :param beta_comp: lognormal standard deviation of the distribution of the dmrs
    :param beta_rtr: record-to-record uncertainty of each fragility
    :param p: percentile of interest for the pth percentile of the risk

    :return: dictionary of the relevant parameters for the fragility distribution
    '''

    beta_tot = np.sqrt(beta_dmr ** 2 + beta_rtr ** 2).round(2)
    dmr_p = stats.lognorm(beta_dmr, scale=dmr_median).ppf(1 - p).round(1)

    return {"dmr_median": dmr_median, "dmr_p": dmr_p, "beta_tot": beta_tot, "beta_rtr": beta_rtr, "beta_dmr": beta_dmr}


def calc_str_damage_risk_stats(fragility_parameters, sa_design, hcurve, imtl, risk_duration=50):
    ''' calculate statistics of the risk distribution, based on design inputs and the hazard curve

    :param fragility_parameters: dictionary of the relevant parameters for the fragility distribution
    :param sa_design: design value for amenity limit state, DIL1
    :param hcurve: hazard curve annual probabilities of exceedance
    :param imtl: hazard curve intensity measures
    :param risk_duration: length of time considered (e.g., risk in 50 years)

    :return: list of risk values - mean, median, and percentile of interest (based on fragility_parameters)
    '''

    dmr_median = fragility_parameters['dmr_median']
    dmr_p = fragility_parameters['dmr_p']
    beta_tot = fragility_parameters['beta_tot']
    beta_rtr = fragility_parameters['beta_rtr']

    risk_metrics = []

    for i, (dmr, beta) in enumerate(zip([dmr_median, dmr_median, dmr_p], [beta_tot, beta_rtr, beta_rtr])):
        median = dmr * sa_design
        risk, _ = risk_convolution(hcurve, imtl, median, beta)
        risk_metrics.append(prob_in_n_years(risk, risk_duration))

    return risk_metrics


def deterministic_sampling(fragility_parameters, n_samples):
    """ sample from a fragility distribution at equal intervals

    """
    dp = 1 / (n_samples + 1)
    p_samples = np.arange(dp, 1, dp)

    comp_median = fragility_parameters['comp_median']
    beta_comp = fragility_parameters['beta_comp']
    sampled_medians = stats.lognorm(beta_comp, scale=comp_median).ppf(p_samples)

    return sampled_medians