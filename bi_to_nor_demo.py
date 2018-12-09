import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, pi
from coinflipviz import  autoformat
from pytexit import py2tex

def plot_bi_nor(df, fontsize=10, mu=0, sigma=1, C='1/(sigma*sqrt(2*pi))', E='-(((X-mu)/sigma)**2)/2', xstepsize=10):
    """
    Given the dataframe with x, n(x), p(x) this provides one plot:
    x vs p(x) along with normal approx curve
    """
    fig, ax1 = plt.subplots(1, 1, figsize=(12,5))
    
    X = df['x'].tolist()
    N = df['n(x)'].tolist()    
    P = df['p(x)'].tolist()    

    ax1.bar(X, P, color="C0")    
    xlabel = 'No of Heads'
    ylabel = 'Probability of Sequences\nhaving those no of Heads'   
    #autoformat(ax1, xlabel, ylabel, fontsize)  
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    
    # standard guassian continuous distribution
    X = np.linspace(-max(X),max(X),100)
    C = eval(C)
    E = eval(E)
    G = C*np.exp(E)
    ax1.plot(X,G, color='red')
    #ax1.set_xlim([-max(X),max(X)])

    #xstepsize = 10
    start, end = ax1.get_xlim()
    ax1.xaxis.set_ticks(np.arange(int(start), int(end), xstepsize))

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.45)    
    plt.show()

def plot_bi_nor_2(df, outcomes, n_events, fontsize=10, ax1=None, ax2=None, 
    C='1/(sigma*sqrt(2*pi))', E='-(((X-mu)/sigma)**2)/2', 
    xstepsize=10, n_shift=False, n_mu=0,
    xlabel = 'X', ylabel1='n (X)', ylabel2='p (X)'):
    """
    Given the dataframe with x, n(x), p(x) this provides two plots:
    x vs n(x) along with normal approx curve
    x vs p(x) along with normal approx curve

    n_shift - theoretical n(x) is probability neutral, so use this switch to shift its normal approx.curve
    when you have unequal probabilities between outcomes.Do not forget to provide new mu value if u enable this.
    """
    mu = 0
    sigma = 1
    mu, var, sigma = get_metrics(df)
    total_outcomes = sum(outcomes)
    p = round(mu/(n_events*total_outcomes),4)
    #print(p)
    #print('Mean: {}  SD: {}'.format(mu, sigma))

    if ax1 == None or ax2 == None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))
    
    X = df['x'].tolist()
    N = df['n(x)'].tolist()    
    P = df['p(x)'].tolist()    

    # FREQUENCY GRAPH    

    ax1.bar(X, N, color="C0")    
    autoformat(ax1, xlabel, ylabel1, fontsize+3)  


    # PROBABILITY GRAPH    
    
    ax2.bar(X, P)    
    autoformat(ax2, xlabel, ylabel2, fontsize+3)      

    # standard normal continuous distribution approxmiation for frequency graph
    X = np.linspace(-max(X),max(X),10*len(X))
    #C = eval(C)
    Cf = df['n(x)'].max()  # max frequency
    if n_shift == True:     # disabling mu influence
        mu_temp = mu
        mu = n_mu
        Ef = eval(E)
        p_temp = round(mu/(n_events*total_outcomes),4)        
        metrics_text = '$\mu_x:{}$ \n$\sigma_x:{}$ \n$p_y:{}$'.format(mu, sigma,p_temp)
        #metrics_text = '$\mu:{}$ \n$\sigma:{}$'.format(mu, sigma)
        font_color='blue'
        mu = mu_temp
    else:
        Ef = eval(E)
        #p = round(mu/(n_events*total_outcomes),4)
        metrics_text = '$\mu_x:{}$ \n$\sigma_x:{}$ \n$p_y:{}$'.format(mu, sigma,p)
        #metrics_text = '$\mu:{}$ \n$\sigma:{}$'.format(mu, sigma)
        font_color='red'
    G = Cf*np.exp(Ef)
    ax1.plot(X,G, color='red')

    ax1.text(0.025,0.98,metrics_text, ha='left', va='top',transform = ax1.transAxes,fontsize=fontsize+3,color=font_color)

    pdf_latex = py2tex('max(n(x))'+ '*(e)**(' + E + ')', print_latex=False, print_formula=False)[1:-1]
    ax1.text(0.97, 0.98,pdf_latex,ha='right', va='top',transform = ax1.transAxes,fontsize=fontsize+5,color='red')    


    # standard normal continuous distribution apprxomation for probability graph
    #X = np.linspace(-max(X),max(X),100)
    Cp = eval(C)
    Ep = eval(E)
    G = Cp*np.exp(Ep)
    ax2.plot(X,G, color='red')


    metrics_text = '$\mu_x:{}$ \n$\sigma_x:{}$ \n$p_y:{}$'.format(mu, sigma,p)
    #metrics_text = '$\mu:{}$ \n$\sigma:{}$'.format(mu, sigma)
    ax2.text(0.025,0.98,metrics_text, ha='left', va='top',transform = ax2.transAxes,fontsize=fontsize+3,color='red')
    pdf_latex = py2tex('(' + C + ')*(e)**(' + E + ')', print_latex=False, print_formula=False)[1:-1]
    ax2.text(0.97, 0.98,pdf_latex,ha='right', va='top',transform = ax2.transAxes,fontsize=fontsize+5,color='red')

    
    # BOTH GRAPHS

    # fix x axis steps and limits for both graphs
    pend = mu + 5*sigma
    pstart = mu - 5*sigma
    if n_shift == True:
        nend = n_mu + 5*sigma
        nstart = n_mu - 5*sigma
        ax1.set_xlim([nstart,nend])
        ax1.xaxis.set_ticks(np.arange(int(nstart), int(nend), xstepsize))    
    else:
        ax1.set_xlim([pstart,pend])
        ax1.xaxis.set_ticks(np.arange(int(pstart), int(pend), xstepsize))    

    ax2.set_xlim([pstart,pend])
    ax2.xaxis.set_ticks(np.arange(int(pstart), int(pend), xstepsize))    

    if ax1 == None or ax2 == None:
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.45)    
        plt.show()  

    #return 

def get_metrics(df):
    """
    given dataframe with x, n(x), p(x) it returns mu, var, sigma
    """
    mu = round((df['x']*df['p(x)']).sum(),4)
    var = round((    ((df['x'] - mu)**2)    *df['p(x)']).sum(),4)
    sigma = round(sqrt(var),4)    
    #n = (df['n(x)']).sum()
    return mu, var, sigma

def plot_summary(theor_df,stats_df, outcomes, n_events, n_experiments, n_mu=0, n_shift=False):
    # normal approximation for probability curve
    C='1/(sigma*sqrt(2*pi))'
    E='-1/2*((X-mu)/sigma)**2'

    # plot the result with normal approximation curves as well
    #if axarr != [] or len(axarr) != 2 or len(axarr[0]) != 2:
    fig, axarr = plt.subplots(2, 2, figsize=(14,10))

    plot_bi_nor_2(theor_df, outcomes, n_events, ax1=axarr[0, 0], ax2=axarr[0, 1], C=C,E=E ,xstepsize=5, n_shift=n_shift, n_mu=n_mu)    # binomial theoretical
    plot_bi_nor_2(stats_df, outcomes, n_events, ax1=axarr[1, 0], ax2=axarr[1, 1], C=C,E=E ,xstepsize=5, 
                xlabel='$\hat{X}$', ylabel1='$n(\widehat{X})$', ylabel2='$p(\widehat{X})$')    # binomial statistical

    # name the charts
    names = [ ['A','B'],['C','D']]
    for i in range(0,len(names)):
        for j in range(0,len(names[0])):
            axarr[i, j].text(0.97, 0.05,names[i][j],ha='right', va='bottom',transform = axarr[i, j].transAxes,fontsize=17.5,color='blue')    
            
    # titles
    fontsize = 14
    axarr[0, 0].set_title('Theoretical Frequency $n(X_k)$', fontsize=fontsize)
    axarr[0, 1].set_title('Theoretical Probability Distribution $p(X_k)$', fontsize=fontsize)
    axarr[1, 0].set_title('Statistical Frequency $n(X_k)$', fontsize=fontsize)
    axarr[1, 1].set_title('Statistical Probability Distribution $p(X_k)$', fontsize=fontsize)
            
    # if axarr != [] or len(axarr) != 2 or len(axarr[0]) != 2:
    plt.tight_layout()
    plt.show()  
    # else:
    #     return axarr  

def bare_minimal_plot(ax, df, mu, sigma, p,
    title='', xlabel='', ylabel='', pstart=-5, pend=25, ymax=0.5,
    norm_off = False):    
    # binomial discrete
    X = df['x'].tolist()
    P = df['p(x)'].tolist()    
    ax.bar(X, P, color="C0") 
    
    # normal continuous
    X = np.linspace(-max(X),max(X),10*len(X))
    Cp = 1/(sigma*sqrt(2*pi))
    Ep = -(((X-mu)/sigma)**2)/2
    G = Cp*np.exp(Ep)    
    if norm_off == False:  # so user wants normal curve overlay
        ax.plot(X, G, color='red')

    # fix x and y to view animation on single scale
    # pend = 25
    # pstart = -5
    ax.set_xlim([pstart,pend])
    ax.set_ylim([0,ymax])
    ax.set_title(title)
    metrics_text = '$\mu_x:{}$ \n$\sigma_x:{}$ \n$p_y:{}$'.format(mu, sigma,p)
    ax.text(0.025,0.98,metrics_text, ha='left', va='top',transform = ax.transAxes,fontsize=13,color='red')



