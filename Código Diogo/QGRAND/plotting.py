import numpy as np
from scipy.special import comb
from scipy.interpolate import interp1d
import sys

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.ticker import ScalarFormatter

from .statistics import get_uniform_at_random_statistics

plt.rc('text',usetex=True)
plt.rc('font', family='serif', size=24)
plt.rcParams['figure.figsize'] = [10, 6]

def plot_infidelity(n,
                    k_list,
                    num_gates,
                    p,
                    max_t,
                    infidelity,
                    confidence_interval = 0.8,
                    interpolation = 'linear',
                    add_subplot_labeling = '(a)',
                    subplot_index = 0,
                    set_linear_formatter = False,
                    fig_ax = None,
                    show = False,
                    save = False):
    """Plots the infidelity vs. code rate.

    Parameters
    ----------
    n : int
        Total number of encoding qubits.
    k_list : 1D array
        1D array of different numbers of data qubits.
    num_gates : 1D array
        List for the different number of gates considered. If the list has 
        length 1, the gate number is omitted from the plots.
    p : 1D array
        Different cases for the probability of error.
    max_t : int
        Maximum weight to be plotted.
    failure_rate : 
        Data array.
    confidence_interval : float, None, default 0.8
        Whether to consider a confidence interval for the data. If ``None``, 
        the confidence interval is not shown. If a float between 0 and 1, the 
        confidence interval is shown.
    interpolation : str, default 'linear'
        Whether to interpolate the points to be shown. Default is 'linear', 
        which is equivalent to not interpolating, as it does not change the 
        plot's appearance. In general, `interpolation` is passed as the `kind`
        keyword argument in the function `scipy.interpolate.interp1d()`.
    add_subplot_labeling : str, default '(a)'
        What labeling to add to the top left corner of the plot. For no 
        labeling, use ``''``.
    set_linear_formatter : bool, list, default False
        By default (``False`` case), a log scale is used for the y axis. To 
        use a linear scale instead, pass a list with the values that should be
        shown in the axis labeling.
    fig_ax : default None
        (fig, ax) Matplotlib figure and axes to use.
    show : bool, default False
        Whether to show the plot.
    save : bool, str, default False
        Whether to save the plot. Does not save it, by default. If ``True``, 
        saves the plot and encodes a timestamp in the filename. If a specific 
        timestamp is passed, it uses that instead.

    Returns
    -------
    fig : optional
        Matplotlib figure.
    ax : optional
        Matplotlib axes.    
    """

    
    conf = (1 - confidence_interval) / 2

    code_rate = np.array(k_list)/n
    x = np.linspace(code_rate[0], code_rate[-1], 100)
    _, failure_theory, any_error_prob = get_uniform_at_random_statistics(n,x,p,max_t)

    # E_N = np.array([(3**(i+1) * comb(n, i+1)) / 2**(n*(1-x)) for i in range(max_t)])
    # B_N = np.insert(np.cumsum(E_N, axis=0)[:-1], 0, 1/ 2**(n*(1-x)), 0)
    # y_theory = np.where(E_N < 1e-3, 1, (1-np.exp(-E_N))/E_N) * np.exp(-B_N)

    # dd = np.arange(1, max_t+1)[:,None, None]
    # failure_theory = any_error_prob - np.cumsum(p**dd * (1-p)**(n-dd) * comb(n, dd) * y_theory[:,:,None], axis=0)

    shade = True if confidence_interval is not None else False

    if fig_ax is None:
        rows = 1
        columns = 1
        fig, ax = plt.subplots(rows,
                            columns,
                            sharex='col',
                            sharey=False,
                            squeeze=False,
                            figsize=(10*columns, 6*rows))
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.05, wspace=0.02)
    else:
        fig, ax = fig_ax

    # Create axes
    ax1 = ax[subplot_index, 0]
    ax2 = ax1.twinx() if p.size > 1 else None
    
    # Add subplot labeling
    trans = mtransforms.ScaledTranslation(-65/72, -25/72, fig.dpi_scale_trans)
    ax1.text(0.0, 1.0, add_subplot_labeling, transform=ax1.transAxes + trans,
                        fontsize='medium', va='bottom', fontfamily='serif')

    # Plot uncoded infidelity
    ax1.plot(code_rate,
        np.linspace(any_error_prob[0], any_error_prob[0], code_rate.shape[0]),
        'k',
        label='uncoded')
    ax2.plot(code_rate,
        np.linspace(any_error_prob[1], any_error_prob[1], code_rate.shape[0]),
        'k--') if p.size > 1 else None

    line = ['.',':','--','-.','-']
    plot = [None]*max_t

    for j, n_gates in enumerate(num_gates):
        for i in range(max_t):# + [n-1]:

            y = interp1d(code_rate, 
                        np.median(infidelity[:, j, :, i, 0], axis=1),
                        kind=interpolation)(x)
            q_m = interp1d(code_rate,
                        np.quantile(infidelity[:, j, :, i, 0], conf, axis=1),
                        kind=interpolation)(x)
            q_p = interp1d(code_rate,
                    np.quantile(infidelity[:, j, :, i, 0], 1-conf, axis=1),
                    kind=interpolation)(x)
            
            label = 'up to $t={}$'.format(i+1) if i<n-1 else 'all corrected'
            
            plot[i] = ax1.plot(x,
                y,
                line[j],
                label=label,
                color=plot[i][-1].get_color() if plot[i] is not None else None
            )
            ax1.fill_between(x,
                            q_m,
                            q_p,
                            alpha=0.2,
                            color=plot[i][-1].get_color()) if shade else None
            ax1.plot(x[::5],
                    failure_theory[i,::5,0],
                    'o',
                    markerfacecolor='none',
                    color=plot[i][-1].get_color()) if j==0 else None

            if p.size > 1:

                y = interp1d(code_rate,
                            np.median(infidelity[:, j, :, i, 1], axis=1),
                            kind=interpolation)(x)
                q_m = interp1d(code_rate,
                        np.quantile(infidelity[:, j, :, i, 1], conf, axis=1),
                        kind=interpolation)(x)
                q_p = interp1d(code_rate,
                    np.quantile(infidelity[:, j, :, i, 1], 1-conf, axis=1),
                    kind=interpolation)(x)
                
                label='up to $t={}$'.format(i+1) if i<n-1 else 'all corrected'
                
                ax2.plot(x, y, '--', color=plot[i][-1].get_color())
                ax2.fill_between(x,
                            q_m,
                            q_p,
                            alpha=0.2,
                            color=plot[i][-1].get_color()) if shade else None
                ax2.plot(x[::5],
                        failure_theory[i,::5,1],
                        'o',
                        markerfacecolor='none',
                        color=plot[i][-1].get_color()) if j==0 else None

    if num_gates.size > 1:
        for j, n_gates in enumerate(num_gates):
            ax1.plot(x,
                    np.zeros_like(x)+any_error_prob[0],
                    'k'+line[j],
                    label='{} gates'.format(n_gates))

    ax1.plot([-0.1],
            [any_error_prob[0]],
            'ko',
            markerfacecolor='none',
            label='uniform')

    # Labeling and legend
    #ax1.set_xlabel('code rate $R$, for $n = {}$'.format(n))
    ax1.set_ylabel('BLER, $p={}$, solid'.format(p[0]))
    ax2.set_ylabel(
        'BLER, $p={}$, dashed'.format(p[1])
    ) if p.size > 1 else None
    ax1.legend(loc='lower right',
                borderpad=0.2,
                labelspacing=0.15,
                handlelength=1,
                handletextpad=0.3,
                borderaxespad=0.2)
    ax1.set_yscale('log')
    ax2.set_yscale('log') if p.size > 1 else None
    ax1.set_xlim([code_rate[0], code_rate[-1]])
    if set_linear_formatter:
        ax1.set_yticks(set_linear_formatter)
        ax1.yaxis.set_major_formatter(ScalarFormatter())
    #ax2.set_ylim([5e-3, None])
    #ax1.autoscale(axis='x', tight=True)
    #plt.show()

def plot_correctable_fraction(n,
                            k_list,
                            num_gates,
                            p,
                            max_t,
                            correctable_fraction,
                            confidence_interval = 0.8,
                            interpolation = 'linear',
                            add_subplot_labeling = '(a)',
                            subplot_index = 0,
                            set_linear_formatter = False,
                            fig_ax = None,
                            show = False,
                            save = False):

    conf = (1 - confidence_interval) / 2

    code_rate = np.array(k_list)/n
    x = np.linspace(code_rate[0], code_rate[-1], 100)
    y_theory, _, _ = get_uniform_at_random_statistics(n,x,p,max_t)

    if fig_ax is None:
        rows = 1
        columns = 1
        fig, ax = plt.subplots(rows,
                            columns,
                            sharex='col',
                            sharey=False,
                            squeeze=False,
                            figsize=(10*columns, 6*rows))
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.05, wspace=0.02)
    else:
        fig, ax = fig_ax

    ax3 = ax[subplot_index,0]

    # Add subplot labeling
    trans = mtransforms.ScaledTranslation(-65/72, -25/72, fig.dpi_scale_trans)
    ax3.text(0.0, 1.0, add_subplot_labeling, transform=ax3.transAxes + trans,
            fontsize='medium', va='bottom', fontfamily='serif')

    line = ['.',':','--','-.','-']
    plot = [None]*max_t
    shade = True if confidence_interval is not None else False

    for j, n_gates in enumerate(num_gates):
        for i in range(max_t):
            if np.mean(correctable_fraction[0, j, :, i]) < 1e-3:
                break
            y = interp1d(code_rate, np.median(correctable_fraction[:, j, :, i], axis=1), kind=interpolation)(x)
            q_m = interp1d(code_rate, np.quantile(correctable_fraction[:, j, :, i], conf, axis=1), kind=interpolation)(x)
            q_p = interp1d(code_rate, np.quantile(correctable_fraction[:, j, :, i], 1-conf, axis=1), kind=interpolation)(x)
            plot[i] = ax3.plot(x, y, line[j], label='$t={}$'.format(i+1) if j==4 else None, color=plot[i][-1].get_color() if plot[i] is not None else None)
            ax3.fill_between(x, q_m, q_p, alpha=0.2, color=plot[i][-1].get_color()) if shade else None
            ax3.plot(x, y_theory[i], 'o', markerfacecolor='none', color=plot[i][-1].get_color()) if j==1 else None

    if num_gates.size > 1:
        for j, n_gates in enumerate(num_gates):
            ax3.plot(x,
                    np.zeros_like(x)-0.1,
                    'k'+line[j],
                    label='{} gates'.format(n_gates))
    ax3.plot(x, x**0+0.1, 'ko', markerfacecolor='none', label='uniform')
    ax3.set_ylabel('$f$')
    ax3.set_xlabel('$R$')
    ax3.legend(loc='upper right',borderpad=0.2, labelspacing=0.15, handlelength=1, handletextpad=0.3, borderaxespad=0.2)
    ax3.set_ylim([0, 1])
    ax3.autoscale(axis='x', tight=True)