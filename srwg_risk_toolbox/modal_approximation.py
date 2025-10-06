import numpy as np

from functools import partial
from scipy import optimize

from srwg_risk_toolbox.structural_system_assumptions import set_alpha

def alpha_response_profiles(alphas, n_stories, eff_height, sdof):
    n_alphas = len(alphas)
    
    T1 = 1
    n_modes = 1
    story_height = 1
    n_floors = n_stories + 1

    sdof = 1
    
    modal_disp_profile = np.zeros([n_alphas, n_modes, n_floors, 1])
    modal_drift_profile = np.zeros([n_alphas, n_modes, n_stories, 1])

    for k,alpha in enumerate(alphas):
        [T, Gamma_phi] = modal_participation(T1, alpha, n_modes, n_stories)

        ## calculate response for each mode
        for i in range(n_modes):
            # include mode shape and participation factor for each floor
            for j in range(n_floors):
                modal_disp_profile[k, i, j, :] = Gamma_phi[i, j] * sdof

            modal_drift_profile[k, i, :, :] = (modal_disp_profile[k, i, 1:, :] - modal_disp_profile[k, i, :-1, :]) / story_height
            
    modal_profiles = {}
    modal_profiles['disp'] = np.squeeze(modal_disp_profile)
    modal_profiles['drift'] = np.squeeze(modal_drift_profile)
    
    return modal_profiles


def plot_drift_profiles(plot_parameters):
    '''
    Plot the drift profiles, including the location on the figure
    '''
    
    # extract parameters
    structural_systems = plot_parameters['structural_systems']
    alphas = [set_alpha(str_sys) for str_sys in structural_systems]
    
    ax = plot_parameters['ax']
    x_min = plot_parameters['x_min']
    x_frac = plot_parameters['x_frac']
    y_min = plot_parameters['y_min']
    y_frac = plot_parameters['y_frac']
    anno_fontsize = plot_parameters['anno_fontsize']
    
    # basic assumptions
    n_stories = 100
    eff_height = n_stories * plot_parameters['eff_height_ratio']
    sdof = 1
    eff_drift = sdof / eff_height
    Ccr_drift = plot_parameters['C_cr'] / eff_height

    # create profiles
    response_profiles = alpha_response_profiles(alphas, n_stories, eff_height, sdof)   
    drift_profiles = response_profiles['drift']

    # scale for the figure
    xlim = ax.get_xlim()
    x_scale = (x_frac * (xlim[1]-xlim[0])) / np.max(drift_profiles)
    x_min *= (xlim[1]-xlim[0])
    drift_profiles *= x_scale
    drift_profiles += x_min
    xlim = [x_min,np.max(drift_profiles)]

    ylim = ax.get_ylim()
    stories = np.arange(n_stories) + 0.5
    y_scale = (y_frac * (ylim[1]-ylim[0])) / n_stories
    y_min *= (ylim[1]-ylim[0])
    stories *= y_scale
    stories += y_min
    ylim = [y_min,stories[-1]]

    eff_drift *= x_scale
    eff_drift += x_min

    Ccr_drift *= x_scale
    Ccr_drift += x_min

    eff_height *= y_scale
    eff_height += y_min

    for k,alpha in enumerate(alphas):
        color = 'dimgray'
        ls = plot_parameters['ls'][k]
        lw = plot_parameters['lw'][k]
        
        _ = ax.plot(drift_profiles[k,:],stories,ls=ls,lw=lw,color=color)

    color='k'
    _ = ax.scatter(eff_drift,eff_height,color=color)
    _ = ax.plot([eff_drift]*2,ylim,color=color,ls=':',zorder=-5)
    _ = ax.plot([eff_drift] * 2, ylim, color=color, ls=':', zorder=-5)
    _ = ax.plot([Ccr_drift] * 2, ylim, color=color, ls=':', zorder=-5)

    color = 'silver'
    _ = ax.plot([x_min]*2,ylim,color=color)
    _ = ax.plot(xlim,[y_min]*2,color=color)
    
    if True:
        # _ = ax.text(xlim[0],ylim[1],'Height',rotation=90,ha='right',va='top')
        _ = ax.text(xlim[1],ylim[0],'\nStorey drift ratio',rotation=0,ha='right',va='center')

        y_factor = 1.1
        x_gap = 5e-4 * x_scale
        _ = ax.text(eff_drift-x_gap, y_factor*ylim[1],'$drift_{effective}$',rotation=90,ha='right',va='top',fontsize=anno_fontsize)
        _ = ax.text(Ccr_drift-x_gap, y_factor*ylim[1], '$drift_{critical}$', rotation=90, ha='right', va='top',fontsize=anno_fontsize)


        for k,str_sys in enumerate(structural_systems):
            if str_sys == 'rc frame':
                label = 'Frame'
                va = 'center'
            elif str_sys == 'wall':
                label = 'Wall'
                va = 'top'
            x = np.max(drift_profiles[k,:])
            y = stories[np.argmax(drift_profiles[k,:])]

            if plot_parameters['annotate_C_cr']:
                _ = ax.text(x,y,' %s\n $C_{cr}$:%s'%(label,plot_parameters['C_cr'][str_sys](0)),va=va)
            
            
def plot_disp_profiles(plot_parameters):
    ''' Plots the displacement profiles, including the position on the figure
    '''
    
    # extract parameters
    structural_systems = plot_parameters['structural_systems']
    alphas = [set_alpha(str_sys) for str_sys in structural_systems]
    
    ax = plot_parameters['ax']
    x_min = plot_parameters['x_min']
    x_frac = plot_parameters['x_frac']
    y_min = plot_parameters['y_min']
    y_frac = plot_parameters['y_frac']
    anno_fontsize = plot_parameters['anno_fontsize']
    
    # basic assumptions
    n_stories = 100
    eff_height = n_stories * plot_parameters['eff_height_ratio']
    sdof = 1
    eff_drift = sdof / eff_height

    # create profiles
    response_profiles = alpha_response_profiles(alphas, n_stories, eff_height, sdof)   
    disp_profiles = response_profiles['disp']

    # scale for the figure
    xlim = ax.get_xlim()
    x_scale = (x_frac * (xlim[1]-xlim[0])) / np.max(disp_profiles)
    x_min *= (xlim[1]-xlim[0])
    disp_profiles *= x_scale
    disp_profiles += x_min
    xlim = [x_min,np.max(disp_profiles)]

    ylim = ax.get_ylim()
    stories = np.arange(n_stories+1).astype('float')
    y_scale = (y_frac * (ylim[1]-ylim[0])) / n_stories
    y_min *= (ylim[1]-ylim[0])
    stories *= y_scale
    stories += y_min
    ylim = [y_min,stories[-1]]

    sdof *= x_scale
    sdof += x_min

    eff_height *= y_scale
    eff_height += y_min

    for k,alpha in enumerate(alphas):
        color = 'dimgray'
        ls = plot_parameters['ls'][k]
        lw = plot_parameters['lw'][k]
        
        _ = ax.plot(disp_profiles[k,:],stories,ls=ls,lw=lw,color=color)

    color='k'
    _ = ax.scatter(sdof,eff_height,color=color)
    _ = ax.plot([x_min,sdof],[y_min,eff_height],color=color,ls='--',zorder=-5)
    _ = ax.plot([sdof] * 2, [y_min,eff_height], color=color, ls=':', zorder=-5)

    color = 'silver'
    _ = ax.plot([x_min]*2,ylim,color=color)
    _ = ax.plot(xlim,[y_min]*2,color=color)
    
    if True:
        # _ = ax.text(xlim[0],ylim[1],'Height',rotation=90,ha='right',va='top')
        _ = ax.text(xlim[1],ylim[0],'\nDisplacement',rotation=0,ha='right',va='center')

        _ = ax.text(sdof, ylim[0], ' $SDOF_{disp}$', rotation=90, ha='right', va='bottom',fontsize=anno_fontsize)
        
        for k,str_sys in enumerate(structural_systems):
            if str_sys == 'rc frame':
                label = 'Frame'
                va = 'center'
            elif str_sys == 'wall':
                label = 'Wall'
                va = 'top'
            x = np.max(disp_profiles[k,:])
            y = stories[np.argmax(disp_profiles[k,:])]


def plot_building_approximation(plot_parameters,structural_systems):
    '''
    Plot the concept of the building as an SDOF
    '''
    from matplotlib.lines import Line2D

    ax = plot_parameters['ax']
    x_min = plot_parameters['x_min']
    x_frac = plot_parameters['x_frac']
    y_min = plot_parameters['y_min']
    y_frac = plot_parameters['y_frac']
    anno_fontsize = plot_parameters['anno_fontsize']

    x_sdof = plot_parameters['x_sdof']
    n_stories = 100
    eff_height = n_stories * plot_parameters['eff_height_ratio']

    bldg_min = plot_parameters['bldg_min']
    bldg_max = plot_parameters['bldg_max']
    bldg_center = np.mean([bldg_min, bldg_max])

    legend_x = 0.1975
    legend_y = 0.4

    # scale for the figure
    xlim = ax.get_xlim()
    x_min *= (xlim[1] - xlim[0])
    x_scale = (x_frac * (xlim[1] - xlim[0]))
    xlim = [x_min, x_min + x_scale]

    ylim = ax.get_ylim()
    stories = np.arange(n_stories) + 0.5
    y_scale = (y_frac * (ylim[1] - ylim[0])) / n_stories
    y_min *= (ylim[1] - ylim[0])
    stories *= y_scale
    stories += y_min
    ylim = [y_min, stories[-1]]

    eff_height *= y_scale
    eff_height += y_min

    x_sdof *= x_scale
    x_sdof += x_min

    bldg_min *= x_scale
    bldg_min += x_min

    bldg_max *= x_scale
    bldg_max * + x_frac

    color = 'k'
    _ = ax.scatter(x_sdof, eff_height, color=color)
    _ = ax.plot([x_sdof] * 2, [ylim[0], eff_height], color=color, ls='-', zorder=-5)

    color = 'silver'
    _ = ax.plot([x_min] * 2, ylim, color=color)
    _ = ax.plot(xlim, [y_min] * 2, color=color)

    color = 'lightgray'
    _ = ax.fill_between([bldg_min, bldg_max], ylim[0], [ylim[1]] * 2, color=color)

    if True:
        _ = ax.text(xlim[0], ylim[1], 'Height', rotation=90, ha='right', va='top')
        _ = ax.text(xlim[1], ylim[0], '\nSDOF approximation', rotation=0, ha='right', va='center')

        x_gap = 1e-2 * x_scale
        _ = ax.text(x_sdof - x_gap, eff_height, '$height_{effective}$  ', rotation=90, ha='right', va='top',
                    fontsize=anno_fontsize)

        anno = '$N_{stories}$'
        _ = ax.annotate(anno, [legend_x, legend_y + 0.35], xycoords='axes fraction', ha='center', va='center',
                        fontsize=anno_fontsize)

        handles = []
        for k, str_sys in enumerate(structural_systems):
            color = 'dimgray'
            ls = plot_parameters['ls'][k]
            lw = 2
            #             lw = plot_parameters['lw'][k]
            if str_sys == 'rc frame':
                label = 'Frame'
            elif str_sys == 'wall':
                label = 'Wall'
            handles.append(Line2D([0], [0], color=color, lw=lw, ls=ls, label=label))

        title = 'Structural\nSystem'
        legend = ax.legend(handles=handles, title=title, handlelength=1.3, bbox_to_anchor=[legend_x, legend_y],
                           loc='center')
        legend.get_title().set_multialignment('center')


def characteristic_eqn(gamma,alpha):
    
    '''
    Charateristic equation for finding the eigenvalues (eqn 24 of Taghavi and Miranda 2005)
    '''
    
    return 2 +\
          (2+(alpha**4/(gamma**2*(gamma**2+alpha**2))))*np.cos(gamma)*np.cosh(np.sqrt(alpha**2+gamma**2)) +\
          (alpha**2/(gamma*np.sqrt(alpha**2+gamma**2)))*np.sin(gamma)*np.sinh(np.sqrt(alpha**2+gamma**2))


def phi_eqn(x,gamma,alpha):
    '''
    Returns the mode shape, phi, over the normalized height of the building
    (eqn 22 of Taghavi and Miranda 2005)
    '''
    
    nu = (gamma**2*np.sin(gamma) + gamma*np.sqrt(alpha**2+gamma**2)*np.sinh(np.sqrt(alpha**2+gamma**2))) /\
         (gamma**2*np.cos(gamma) + (alpha**2+gamma**2)*np.cosh(np.sqrt(alpha**2+gamma**2)))
    
    if False:
        # Eqn 22, as published in Taghavi and Miranda 2005
        phi = (np.sin(gamma*x)-gamma*(alpha**2+gamma**2)**(-0.5)*np.sinh(x*np.sqrt(alpha**2+gamma**2))+nu*(np.cosh(x*np.sqrt(alpha**2+gamma**2))-np.cos(gamma*x))) /\
           (np.sin(gamma)  -gamma*(alpha**2+gamma**2)**(-0.5)*np.sinh(  np.sqrt(alpha**2+gamma**2))+nu*(np.cosh(  np.sqrt(alpha**2+gamma**2))-np.cos(gamma)))
    else:
        # altered Eqn 22, per personal communication with Miranda
        phi = (np.sin(gamma*x)-gamma*(alpha**2+gamma**2)**(-0.5)*np.sinh(x*np.sqrt(alpha**2+gamma**2))+nu*(np.cosh(x*np.sqrt(alpha**2+gamma**2))-np.cos(gamma*x)))

    return phi


def find_roots(f,n_roots):
    
    '''
    Finds first n roots of the equation by looking for intervals [a-b]
    where there is a change of sign between f(a) and f(b)
    '''
    
    roots = np.zeros([n_roots])
    
    dx = 0.01
    a = dx
    b = a + dx
    
    # find the ascending roots
    for i in range(n_roots):
        
        # shift the interval until there is a sign change
        while (f(a)>0 and f(b)>0) or (f(a)<0 and f(b)<0):
            a = b
            b = b + dx
        
        # find the root between a and b
        roots[i] = optimize.brentq(f, a, b)
        
        # shift interval to find the next root
        a = b
        
    return roots


def modal_participation(T1,alpha,n_modes,n_stories):
    
    '''
    Computes the modal response of the simplified model
    '''

    ## location of each floor, normalized by building height
    H = 1
    x = np.linspace(0,H,n_stories+1)

    ## preallocate variables
    gamma = np.round(find_roots(partial(characteristic_eqn,alpha=alpha),n_modes),4)
    T = np.zeros_like(gamma)
    Gamma = np.zeros_like(gamma)
    phi = np.zeros([n_modes,len(x)])
    Gamma_phi = np.zeros_like(phi)
    
    ## plot = True shows the mode shapes and Gamma * phi
    plot = False
    if plot:
        fig,ax = plt.subplots(1,2,figsize=(10,4))

    ## compute period, mode shapes, and modal participation factors for each mode
    for i in range(n_modes):

        # period
        if i == 0:
            T[i] = T1
        else:
            # calculate higher mode periods, eqn 25\
            T[i] = T1 * gamma[0]/gamma[i] * np.sqrt((gamma[0]**2+alpha**2)/(gamma[i]**2+alpha**2))

        # mode shape over the height of the building, eqn 22
        phi[i,:] = phi_eqn(x,gamma[i],alpha)

        # modal participation factor, eqn 10
        Gamma[i] = np.trapz(phi[i,:],x) / np.trapz(phi[i,:]**2,x)

        # combined mode shape and participation, Gamma_phi
        Gamma_phi[i,:] = Gamma[i] * phi[i,:]
        
        if plot:
            _ = ax[0].plot(phi[i,:],x)
            _ = ax[1].plot(Gamma_phi[i,:],x)
        
        
    return T, Gamma_phi