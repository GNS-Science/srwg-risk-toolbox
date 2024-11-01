from .base import *

scale_factors_Ccr = {'rc frame':1.4,
                     'wall': 1.55}

def calculate_system_deformation_capacity(masses, displacements):
    
    return np.sum(masses * displacements**2) / np.sum(masses * displacements)

def calculate_effective_mass(masses, displacements):
    
    return np.sum(masses * displacements)**2 / np.sum(masses * displacements**2)

def calculate_effective_stiffness(base_shear, deformation_capacity):
    
    return base_shear / deformation_capacity

def calculate_effective_period(mass_eff, K_eff):
    
    return 2 * np.pi * np.sqrt(mass_eff / K_eff)

def calculate_effective_height(masses, displacements, heights):
    
    return np.sum(masses * displacements * heights) / np.sum(masses * displacements)



def frame_displacement_profile(critical_drift,heights):
    
    n_stories = len(heights)
    
    if n_stories <= 4:
        displacement = critical_drift * heights
    else:
        displacement = critical_drift * heights * ((4*heights[-1]-heights)/(4*heights[-1]-heights[0]))
    
    return displacement


def Sullivan_wall_effective_height_ratio(n_stories):
    # Sullivan (2011) 'An Energy-Factor Method for the DBSD of RC Wall Structures'
    # Eqn. 24

    return 0.7 + ((np.sqrt(n_stories) - 0.7) / n_stories ** 2)


def Sullivan_wall_effective_mass_ratio(n_stories):
    # Sullivan (2011) 'An Energy-Factor Method for the DBSD of RC Wall Structures'
    # Eqn. 23

    return (4 * n_stories - 1) / (3 * n_stories)


def effective_height_ratio_assumptions(structural_system, n_stories):
    if structural_system == 'wall':
        eff_height_ratio = Sullivan_wall_effective_height_ratio(n_stories)

    elif 'frame' in structural_system:
        # placeholder values - result is not sensitive to them
        story_height = 1
        masses = [1] * n_stories
        critical_drift = 1

        heights = np.cumsum([story_height] * n_stories)
        displacements = frame_displacement_profile(critical_drift, heights)
        eff_height = calculate_effective_height(masses, displacements, heights)
        eff_height_ratio = eff_height / heights[-1]

    else:
        raise NameError(structural_system + ' not specified.')

    return eff_height_ratio


def nominal_period_NZS11705(building_height, structural_system, limit_state):
    
    ## based on Commentary to NZS1170.5 period estimation for the serviceability limit state
    
    if structural_system == 'rc frame':
        k_t = 0.075
    elif structural_system == 'wall':
        k_t = 0.05
    elif structural_system == 'steel moment frame':
        k_t = 0.085
    elif structural_system == 'steel braced frame':
        k_t = 0.075
    else:
        raise NameError(structural_system + ' not specified.')
    
    if limit_state == 'uls':
        a = 1.25
    elif limit_state == 'sls':
        a = 1.0
    else:
        error
        
    return a * k_t * building_height ** 0.75


def nominal_period_Pettinga(building_height, structural_system, limit_state):
    
    ## based on Commentary to NZS1170.5 period estimation for the serviceability limit state
    ## Pettinga et al 2019 adds the 1.33 multiplier to amplify the conservatively low estimate
    
    amplifier = 1.33

    return amplifier * nominal_period_NZS11705(building_height, structural_system, limit_state)


def nominal_period_Crowley(building_height):
    ## Crowley and Pinho 2004, see Displacement-based Seismic Design of Structures, p.12

    return 0.1 * building_height

def nominal_period_basic(n_stories):

    return 0.1 * n_stories


def set_alpha(structural_system):

    if 'frame' in structural_system:
        alpha = 20
    elif 'wall' in structural_system:
        alpha = 0
    else:
        raise NameError('Structural system unrecognized.')

    return alpha
