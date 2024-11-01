from .base import *


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