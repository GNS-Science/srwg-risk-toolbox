import numpy as np
from typing import NamedTuple

N_DECIMALS = 2

PRE_ROUND = False
PRE_N_DECIMALS = 3

class SoilProfile(NamedTuple):
    Tc: float
    TL: float
    z_liq: float
    q_c1Ncs_v: float
    q_c1Ncs_sh: float
    q_c1Ncs_cl: float
    Cv_IB: float
    Csh_IB: float
    deposit_type: str

class SiteHazard(NamedTuple):
    pga: float
    m_w: float
    sa1: float
    pga_7pt5: float

class FoundationProfile(NamedTuple):
    DE: float
    B: float
    Q: float


def TL_from_Tc(Tc):
    """Appendix A, Eqn. A2

    Args:
        Tc: thickness of crust below the foundation base

    Returns:
        TL: thickness of liquefiable soil
    """
    TL = 10. - 1. - Tc
    return TL


def msf_from_Mw(m_w):
    """Appendix B, Eqn. B2

    Args:
        m_W: earthquake magnitude

    Returns:
        msf: magnitude scaling factor
    """
    msf = 1. + 0.26 * (8.64 * np.exp(-m_w / 4) - 1.325)
    return msf


def calc_pga_7pt5(pga, m_w, decimal_places=3):
    """Appendix B, Eqn. B1

    Args:
        pga: peak ground acceleration [g]
        m_w: earthquake magnitude

    Returns:
        pga_7pt5: magnitude weighted pga, weighted for Mw = 7.5
    """
    pga_7pt5 = np.round(pga / msf_from_Mw(m_w), decimal_places)
    return pga_7pt5


def calc_Sv_max(q_c1Ncs_v, TL):
    """Dhakal & Cubrinovski  Eqn 13

    Args:
        q_c1Ncs_v:
        TL:

    Returns
        Sv_max:
    """
    Sv_max = 1020 * q_c1Ncs_v**-0.82 * TL
    return Sv_max


def calc_Sv_ratio(pga_7pt5, q_c1Ncs_v):
    """Dhakal & Cubrinovski Eqn 14


    """
    pga_trigger = (1/9) \
                  - (q_c1Ncs_v / 333.3) \
                  + (q_c1Ncs_v / 119)**2 \
                  - (q_c1Ncs_v / 116)**3 \
                  + (q_c1Ncs_v / 147.1)**4

    pga_max = - (1/20) \
                  + (q_c1Ncs_v / 201) \
                  - (q_c1Ncs_v / 156.8)**2 \
                  + (q_c1Ncs_v / 266)**3 \
                  + (q_c1Ncs_v / 175)**4

    if pga_7pt5 <= pga_trigger:
        Sv_ratio = 0
    elif pga_7pt5 >= pga_max:
        Sv_ratio = 1
    else:
        Sv_ratio = (pga_7pt5 - pga_trigger) / (pga_max - pga_trigger)

    return Sv_ratio


def volumetric_settlement(liquefaction_parameters):

    pga_7pt5 = liquefaction_parameters['site_hazard'].pga_7pt5

    TL = liquefaction_parameters['soil_profile'].TL
    q_c1Ncs_v = liquefaction_parameters['soil_profile'].q_c1Ncs_v
    Cv_IB = liquefaction_parameters['soil_profile'].Cv_IB


    Sv_max = calc_Sv_max(q_c1Ncs_v, TL)
    Sv_ratio = calc_Sv_ratio(pga_7pt5, q_c1Ncs_v)

    Sv = Sv_ratio * Sv_max

    Sv_IB = Cv_IB * Sv
    Sv_IB_LB = max(Sv_IB - 25, 0)
    Sv_IB_UB = Sv_IB + 25

    if PRE_ROUND:
        Sv_IB = np.round(Sv_IB, PRE_N_DECIMALS)
        Sv_IB_LB = max(Sv_IB - 25, 0)
        Sv_IB_UB = Sv_IB + 25

        Sv_IB_LB = np.round(Sv_IB_LB, PRE_N_DECIMALS)
        Sv_IB_UB = np.round(Sv_IB_UB, PRE_N_DECIMALS)

    return Sv_IB, (Sv_IB_LB, Sv_IB_UB)


def interpolate_Ssh_max_for_TL(points, TL):

    x = [point[0] for point in points]
    y = [point[1] for point in points]

    return np.interp(TL, x, y)


def calc_Ssh_max(q_c1Ncs_sh, TL):
    """Dhakal & Cubrinovski  Table 1

    Args:
        q_c1Ncs_v:
        TL:

    Returns
        Sv_max:
    """

    Ssh_points = {50: [(0, 0), (1, 40), (7.5, 430), (9, 865)],
                  60: [(0, 0), (1, 40), (7.9, 400), (9, 620)],
                  75: [(0, 0), (1, 40), (8.0, 350), (9, 470)],
                  100: [(0, 0), (1, 40), (5.0, 170), (9, 360)],
                  125: [(0, 0), (1, 40), (5.0, 140), (9, 310)],
                  150: [(0, 0), (1, 40), (6.0, 150), (9, 280)],
                  180: [(0, 0), (1, 0), (6.0, 0), (9, 0)]}

    # assign upper and lower q categories for interpolation
    q_values = list(Ssh_points.keys())
    q_idx = np.searchsorted(q_values, q_c1Ncs_sh, side='right')

    if q_c1Ncs_sh >= 180:
        q_upper = q_lower = 180

    elif q_c1Ncs_sh <= 50:
        q_upper = q_lower = 50

    elif q_c1Ncs_sh in q_values:
        q_upper = q_lower = q_c1Ncs_sh

    else:
        q_upper = q_values[q_idx]
        q_lower = q_values[q_idx - 1]

    # calculate upper and lower bounds for interpolation
    Ssh_max_lower = interpolate_Ssh_max_for_TL(Ssh_points[q_lower], TL)
    Ssh_max_upper = interpolate_Ssh_max_for_TL(Ssh_points[q_upper], TL)

    Ssh_max = np.interp(q_c1Ncs_sh, [q_lower, q_upper], [Ssh_max_lower, Ssh_max_upper])

    return Ssh_max


def calc_Ssh_ratio(pga_7pt5, q_c1Ncs_sh):
    """Dhakal & Cubrinovski Eqn 20, 21


    """
    pga_limit = - (1 / 20) \
                + (q_c1Ncs_sh / 201) \
                - (q_c1Ncs_sh / 156.8) ** 2 \
                + (q_c1Ncs_sh / 266) ** 3 \
                + (q_c1Ncs_sh / 175) ** 4

    Ssh_ratio = min(((500 * pga_7pt5 - 42) / 308), \
                    ((2079 * pga_7pt5 - 829 * pga_limit - 105) / 770)
                    )

    Ssh_ratio = max(Ssh_ratio, 0)
    Ssh_ratio = min(Ssh_ratio, 1)

    return Ssh_ratio


def correction_for_Q(Q):
    CQ = (Q / 250) ** 4.59 * np.exp(-0.42 * (np.log(Q) ** 2 - np.log(250) ** 2))
    return CQ


def correction_for_B(B):
    CB = np.exp((6 - B) / 50)
    return CB


def correction_for_Sa1(sa1, pga_7pt5):
    CSa1 = ((sa1 / pga_7pt5) / 1.5) ** 0.41
    return CSa1


def correction_for_DE(DE, z_liq):
    if DE > z_liq:
        assert 'include calculation for DE > z_liq'
    else:
        CDE = 1
    return CDE


def correction_for_Ssh(liquefaction_parameters):

    pga_7pt5 = liquefaction_parameters['site_hazard'].pga_7pt5
    sa1 = liquefaction_parameters['site_hazard'].sa1

    z_liq = liquefaction_parameters['soil_profile'].z_liq

    Q = liquefaction_parameters['foundation_profile'].Q
    B = liquefaction_parameters['foundation_profile'].B
    DE = liquefaction_parameters['foundation_profile'].DE

    CQ = correction_for_Q(Q)
    CB = correction_for_B(B)
    CSa1 = correction_for_Sa1(sa1, pga_7pt5)
    CDE = correction_for_DE(DE, z_liq)
    Csh = CQ * CB * CSa1 * CDE

    return Csh


def shear_induced_settlement(liquefaction_parameters):

    pga_7pt5 = liquefaction_parameters['site_hazard'].pga_7pt5

    q_c1Ncs_sh = liquefaction_parameters['soil_profile'].q_c1Ncs_sh
    TL = liquefaction_parameters['soil_profile'].TL
    Csh_IB = liquefaction_parameters['soil_profile'].Csh_IB

    Ssh_max = calc_Ssh_max(q_c1Ncs_sh, TL)
    Ssh_ratio = calc_Ssh_ratio(pga_7pt5, q_c1Ncs_sh)

    Ssh = Ssh_ratio * Ssh_max

    Csh = correction_for_Ssh(liquefaction_parameters)

    Ssh_IB = Csh_IB * Csh * Ssh
    Ssh_IB_LB = max(Ssh_IB - 20, 0)
    Ssh_IB_UB = Ssh_IB + 20

    if PRE_ROUND:
        Ssh_IB = np.round(Ssh_IB, PRE_N_DECIMALS)
        Ssh_IB_LB = max(Ssh_IB - 20, 0)
        Ssh_IB_UB = Ssh_IB + 20

        Ssh_IB_LB = np.round(Ssh_IB_LB, PRE_N_DECIMALS)
        Ssh_IB_UB = np.round(Ssh_IB_UB, PRE_N_DECIMALS)

    return Ssh_IB, (Ssh_IB_LB, Ssh_IB_UB)


def ejecta_coverage_equation(pga_7pt5, parameters):
    a, b, max_AeAt = parameters.values()

    AeAt = min( (pga_7pt5 - a) / (b - a) * 100, max_AeAt)

    return AeAt


def calc_ejecta_coverage(pga_7pt5, q_c1Ncs_cl, deposit_type):
    """Dhakal & Cubrinovski, Eqns. 30 - 36

    Args:
        pga_7pt5:
        q_c1Ncs_CL:

    Returns:
        AeAt:
    """
    AeAt_parameters = {'VC': \
                 {100: {'a': 0.17, 'b': 0.25, 'max_AeAt': 100},
                  110: {'a': 0.20, 'b': 0.40, 'max_AeAt': 100},
                  120: {'a': 0.22, 'b': 0.58, 'max_AeAt': 50},
                  130: {'a': 0.25, 'b': 1.00, 'max_AeAt': 20},
                  140: {'a': 0.30, 'b': 1.05, 'max_AeAt': 20},
                  150: {'a': 0.37, 'b': 1.12, 'max_AeAt': 20},
                  180: {'a': pga_7pt5, 'b': 1.00, 'max_AeAt': 0},
                  },
                 'IB': {'a': 0.25, 'b': 1.00, 'max_AeAt': 20}
                 }

    if deposit_type == 'IB':
        parameters = AeAt_parameters[deposit_type]
        AeAt = ejecta_coverage_equation(pga_7pt5, parameters)

    else:
        # assign upper and lower q categories for interpolation
        q_values = list(AeAt_parameters['VC'].keys())
        q_idx = np.searchsorted(q_values, q_c1Ncs_cl, side='right')

        if q_c1Ncs_cl >= 180:
            q_upper = q_lower = 180

        elif q_c1Ncs_cl <= 100:
            q_upper = q_lower = 100

        elif q_c1Ncs_cl in q_values:
            q_upper = q_lower = q_c1Ncs_cl


        else:
            q_upper = q_values[q_idx]
            q_lower = q_values[q_idx - 1]

        lower_parameters = AeAt_parameters['VC'][q_lower]
        upper_parameters = AeAt_parameters['VC'][q_upper]

        AeAt_lower = ejecta_coverage_equation(pga_7pt5, lower_parameters)
        AeAt_upper = ejecta_coverage_equation(pga_7pt5, upper_parameters)

        AeAt = np.interp(q_c1Ncs_cl, [q_lower, q_upper], [AeAt_lower, AeAt_upper])

    AeAt = max(AeAt, 0)

    return AeAt


def ejecta_related_settlement(liquefaction_parameters):

    pga_7pt5 = liquefaction_parameters['site_hazard'].pga_7pt5

    q_c1Ncs_cl = liquefaction_parameters['soil_profile'].q_c1Ncs_cl
    deposit_type = liquefaction_parameters['soil_profile'].deposit_type

    AeAt = calc_ejecta_coverage(pga_7pt5, q_c1Ncs_cl, deposit_type)

    SE = 1.6 * AeAt
    SE_LB = 0.5 * SE
    SE_UB = 1.25 * SE

    if PRE_ROUND:
        SE = np.round(SE, PRE_N_DECIMALS)
        SE_LB = 0.5 * SE
        SE_UB = 1.25 * SE

        SE_LB = np.round(SE_LB, PRE_N_DECIMALS)
        SE_UB = np.round(SE_UB, PRE_N_DECIMALS)

    return SE, (SE_LB, SE_UB)


def ground_settlement_from_sources(liquefaction_parameters):
    Sv, (Sv_LB, Sv_UB) = volumetric_settlement(liquefaction_parameters)
    SE, (SE_LB, SE_UB) = ejecta_related_settlement(liquefaction_parameters)

    SG = np.round(Sv + SE, N_DECIMALS)
    SG_LB = np.round(Sv_LB + SE_LB, N_DECIMALS)
    SG_UB = np.round(Sv_UB + SE_UB, N_DECIMALS)

    return SG, (SG_LB, SG_UB)


def building_settlement_from_sources(liquefaction_parameters):
    Ssh, (Ssh_LB, Ssh_UB) = shear_induced_settlement(liquefaction_parameters)
    Sv, (Sv_LB, Sv_UB) = volumetric_settlement(liquefaction_parameters)
    SE, (SE_LB, SE_UB) = ejecta_related_settlement(liquefaction_parameters)

    SB = np.round(Ssh + Sv + SE, N_DECIMALS)
    SB_LB = np.round(Ssh_LB + Sv_LB + SE_LB, N_DECIMALS)
    SB_UB = np.round(Ssh_UB + Sv_UB + SE_UB, N_DECIMALS)

    return SB, (SB_LB, SB_UB)