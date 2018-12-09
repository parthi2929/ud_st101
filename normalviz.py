import numpy as np
import matplotlib.mlab as mlab
import math

def draw_normal(ax, mu, sigma, cond=''):
    """
    cond: to shade the area meeting the condition
    """
    xstart = mu - 4*sigma
    xend = mu + 4*sigma
    x = np.linspace(xstart, xend, 100)
    y = mlab.normpdf(x, mu, sigma)
    ax.plot(x,y, color='black')
    
    # shade area satisfying the condition
    w = x[eval(cond)] if cond != '' else x
    w_shade = mlab.normpdf(w, mu, sigma)
    ax.fill_between(w, 0, w_shade)
    
    # set x axis in multiples of sigma
    x_ticks = []
    for step in range(-4,5): # 4 sigma on right, 4 on left, mu on middle
        x_tick = round(mu + (step)*sigma,2)
        x_ticks.append(x_tick)        
    ax.xaxis.set_ticks(x_ticks)
    ax.grid(True,  linestyle='--',alpha=0.5)
    
    # symbol x axis
    x_symbols = ['$\mu-4\sigma$','$\mu-3\sigma$','$\mu-2\sigma$','$\mu-\sigma$','$\mu$','$\mu+\sigma$',
                '$\mu+2\sigma$','$\mu+3\sigma$','$\mu+4\sigma$']
    ax_symbols = ax.twiny()
    ax_symbols.xaxis.set_ticks(x_ticks)
    ax_symbols.set_xticklabels(x_symbols)
    ax_symbols.set_xbound(ax.get_xbound())
    
    ax.set_ylim(ymin=0) 