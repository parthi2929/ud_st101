# SDSP = Sample Distribution of Sample Proportions
# This is helper file for programmatic illustrations of SDSP concepts. 

from random import shuffle
import pandas as pd
import numpy as np
from math import sqrt, pi

def create_bernoulli_population(N, p):
    """
    Given the total size of population N, probability of a specific outcome,
    and associated bernoulli variable as list (of outcomes), this returns a shuffled
    population list
    N - Population size, eg N=10000
    p - probability of interested outcome  
    Returns list of 1s and 0s. 1 - indicates the interested outcome, 0 - otherwise
    """
    population_yellow = [1]*(int(p*N))
    population_others = [0]*(int((1-p)*N))
    population = population_yellow + population_others
    shuffle(population)    
    return population

def get_frequency_df(raw_list):
    """
    Given a raw list, this provides frequency of duplicate items along with its probability
    Eg: 
    X  n(X)  p(X)
    0  4000  0.4 
    1  6000  0.6
    If you assume 1 indicates, say a yellow ball, 0 otherwise, then there are 6000 yellow balls
    in given population list, so p(yellow_balls) = 0.6
    """
    # first convert to dictionary of values
    dummy_dict = {i:raw_list.count(i) for i in raw_list}
    freq_dict = {'x':[], 'n(x)':[]}
    freq_dict['x'] = list(dummy_dict.keys())
    freq_dict['n(x)'] = list(dummy_dict.values())

    # dictionary to pd easy transform
    freq_df = pd.DataFrame.from_dict(freq_dict)
    freq_df = freq_df[['x','n(x)']]
    total = freq_df['n(x)'].sum()
    freq_df['p(x)'] = freq_df['n(x)']/total
    freq_df.sort_values('x', inplace=True)
    #freq_df = freq_df.set_index(['x'])  # since cant access by df['x'] this creates compatable issues with my other legacy methods reuse
    return freq_df    

def get_density_df(raw_list, n):
    """
    There are issues in comparing discrete mass with continous density functions, so this function is another helper
    https://math.stackexchange.com/questions/2875762/normal-approximation-breaks-for-sd-0-4/2875790
    """
    # first convert to dictionary of values
    dummy_dict = {i:raw_list.count(i) for i in raw_list}
    freq_dict = {'x':[], 'n(x)':[]}
    freq_dict['x'] = list(dummy_dict.keys())
    freq_dict['n(x)'] = list(dummy_dict.values())

    # dictionary to pd easy transform
    freq_df = pd.DataFrame.from_dict(freq_dict)
    freq_df = freq_df[['x','n(x)']]
    total = freq_df['n(x)'].sum()
    
    freq_df['d(x)'] = n*freq_df['n(x)']/total    # because its density df
    freq_df['p(x)'] = freq_df['n(x)']/total      # also prob as a side kick!
    
    freq_df.sort_values('x', inplace=True)
    return freq_df   

def plot_pdf(df, ax, n_pickups, mu, sigma, p, bar_width=0.05, title='', norm_off = False, dd=[]):
    """
    dd - discrete density functions, sometimes instead of probability discrete function p(x) u may want to use this
    """

    # discrete distribution
    X = df['x'].tolist()
    #X = df.index.tolist()
    if dd == []:  
        Y = df['p(x)'].tolist()  
    else:
        Y = dd  
    ax.bar(X, Y, width=bar_width, color="C0") 

    # normal approximation overlay
    X = np.linspace(0,max(X),10*len(X))
    #sigma = sigma/sqrt(n_pickups)   this did not help
    #print(mu,sigma)
    Cp = 1/(sigma*sqrt(2*pi))
    Ep = -1/2*((X-mu)/sigma)**2
    G = Cp*np.exp(Ep)    
    if norm_off == False:  # so user wants normal curve overlay
        ax.plot(X, G, color='red')    

    # info display
    metrics_text = '$\mu_x:{}$ \n$\sigma_x:{}$ \n$p_y:{}$'.format(mu, sigma,p)
    ax.text(0.025,0.98,metrics_text, ha='left', va='top',transform = ax.transAxes,fontsize=13,color='C0')
    ax.set_title(title)


from random import choices
def sample_for_SDSP(population, n_experiments, n_pickups):
    """
    In given population, for 'n_experiments' times, 
        1. choose sample of size 'n_pickups' from population
        2. find its mean
        3. add that mean to a list
    4. Create density dataframe out of tha list (x, n(x), d(x), p(x))
    5. Return that
    """
    X_hat = []
    X_mean_list = []
    for each_experiment in range(n_experiments):  
        X_hat = choices(population, k=n_pickups)  
        X_mean = sum(X_hat)/len(X_hat)
        X_mean_list.append(X_mean)
    df = get_density_df(X_mean_list, n_pickups)
    return df, X_mean_list


def plot_SDSP(raw_list, axarr, titles=[], width=0.05, index_list=[], norm_off = False,
    bars_color = '#A8D7AF', index_color='#FF5722',
    backgroundColour='#F5F5FF'):
        
    
    # actuality
    axarr[0].hist(raw_list, width=width,  color=bars_color)
    axarr[0].set_facecolor(backgroundColour)

    # density
    _, bins,_ = axarr[1].hist(raw_list, normed=True, width=width, color=bars_color)
    axarr[1].set_facecolor(backgroundColour)
    # density is always most suitable to draw normal approximation
    mu, var, sigma = get_metrics(raw_list)
    X = np.linspace(min(bins),max(bins),10*len(bins))
    Cp = 1/(sigma*sqrt(2*pi))
    Ep = -1/2*((X-mu)/sigma)**2
    G = Cp*np.exp(Ep)    
    if norm_off == False:  # so user wants normal curve overlay
        axarr[1].text(0.97, 0.98,get_normal_curve_label(),ha='right', va='top',transform = axarr[1].transAxes,fontsize=20,color='red')    
        axarr[1].plot(X, G, color='red')  

    # probability
    dummy_dict = {i:raw_list.count(i) for i in raw_list}
    total = sum(list(dummy_dict.values()))
    prob_dict = {key: round(val/total,4) for key, val in dummy_dict.items()}
    axarr[2].bar(list(prob_dict.keys()),list(prob_dict.values()), width=width, color=bars_color)
    axarr[2].set_facecolor(backgroundColour)
        
    # cosmetics
    if len(titles) >= 1:
        axarr[0].set_title(titles[0])
    if len(titles) >= 2:
        axarr[1].set_title(titles[1])
    if len(titles) >= 3:
        axarr[2].set_title(titles[2])
    for j in range(len(axarr)):
        if len(index_list) > j:
            index = index_list[j]
            axarr[j].text(0.97, 0.05,index,ha='right', va='bottom',transform = axarr[j].transAxes,fontsize=25,color=index_color)    

def get_metrics(raw_list):
    """
    given the raw random sample or population as a list, this will provide the metrics of that data
    mu, var, sigma
    """
    dummy_dict = {i:raw_list.count(i) for i in raw_list}
    total = sum(list(dummy_dict.values()))
    # print(list(dummy_dict.keys()))
    # print(total)
    prob_dict = {key: round(val/total,4) for key, val in dummy_dict.items()}
    mean = round(sum(k*v for k,v in prob_dict.items()),4)
    variance = round(sum(((k-mean)**2)*v for k,v in prob_dict.items()),4)
    from math import sqrt
    sd = round(sqrt(variance),4)
    return mean, variance, sd

from pytexit import py2tex
def get_normal_curve_label():
    C='1/(sigma*sqrt(2*pi))'
    E='-(1/2)*(((X-mu)/sigma)**2)'
    pdf_latex = py2tex('(' + C + ')*(e)**(' + E + ')', print_latex=False, print_formula=False)[1:-1]
    #pdf_latex = pdf_latex.replace('$','') # strip to insert some more
    #pdf_latex = '$\\fontsize{30pt}{3em}\\mu$'
    #print(pdf_latex)
    return pdf_latex

def plot_SDSM(raw_list, bins, axarr, titles=[], width=1, index_list=[], norm_off = False,
    bars_color = '#A8D7AF', index_color='#FF5722', backgroundColour = '#F5F5FF'):
        
    
    # actuality
    axarr[0].hist(raw_list, bins, color=bars_color)
    axarr[0].set_facecolor(backgroundColour)

    # density
    _, bins,_ = axarr[1].hist(raw_list, bins, normed=True, color=bars_color)
    axarr[1].set_facecolor(backgroundColour)
    # density is always most suitable to draw normal approximation
    mu, var, sigma = get_metrics(raw_list)
    X = np.linspace(min(bins),max(bins),10*len(bins))
    Cp = 1/(sigma*sqrt(2*pi))
    Ep = -1/2*((X-mu)/sigma)**2
    G = Cp*np.exp(Ep)    
    if norm_off == False:  # so user wants normal curve overlay
        axarr[1].text(0.97, 0.98,get_normal_curve_label(),ha='right', va='top',transform = axarr[1].transAxes,fontsize=20,color='red')    
        axarr[1].plot(X, G, color='red')  

    # probability
    dummy_dict = {i:raw_list.count(i) for i in raw_list}
    total = sum(list(dummy_dict.values()))
    prob_dict = {key: round(val/total,4) for key, val in dummy_dict.items()}
    axarr[2].bar(list(prob_dict.keys()),list(prob_dict.values()), width=width, color=bars_color) #, edgecolor=edgecolor) # edge gives a cascading effect
    axarr[2].set_facecolor(backgroundColour)
        
    # cosmetics
    if len(titles) >= 1:
        axarr[0].set_title(titles[0])
    if len(titles) >= 2:
        axarr[1].set_title(titles[1])
    if len(titles) >= 3:
        axarr[2].set_title(titles[2])
    for j in range(len(axarr)):
        if len(index_list) > j:
            index = index_list[j]
            axarr[j].text(0.97, 0.05,index,ha='right', va='bottom',transform = axarr[j].transAxes,fontsize=25,color=index_color)    

def drawBarGraph(raw_list, ax, popStats, 
    title='Freq. Distribtuion', xlabel='Random Variable', ylabel='Counts',
    xmin = 1):
    """
    draws the population graph for now
    """
    popSize = popStats[0]
    popMean = popStats[1]
    popVar = popStats[2]
    popSigma = popStats[3]

    # create population dataframe 
    import pandas as pd
    columns = ['x', 'freq']
    df = pd.DataFrame(columns=columns)
    for i in range(0, len(raw_list)):
        j = i+1 if xmin==1 else i
        df = df.append({'x': j, 'freq': raw_list[i] }, ignore_index=True) 

    # plot the population graph
    X = df['x'].tolist()
    F = df['freq'].tolist()
    ax.bar(X,F, color='#A8D7AF', edgecolor='#009600')
    
    # make x axis as integers
    
    xmaxint = int(ax.get_xlim()[1])
#     print(xmaxint)
    xint = range(xmin, xmaxint+1)
    ax.set_xticks(xint)    
    
    # cosmetics
    title = title + '\n' + r'$T =  \ {{{0}}}, \ \ \mu = \ {{{1}}}, \ \ \sigma^2 = {{{2}}} \ \ \sigma = {{{3}}}$'.format(popSize, popMean, popVar, popSigma)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize = 14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.yaxis.grid(True, alpha=0.3)
    #ax.set_facecolor(backgroundColour)
    #fig.patch.set_facecolor(backgroundColour)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)    

def getPopulationStatistics(pop, popMin):
    """
    Returns size, mean and variance of given population list
    """
    size = 0
    mean = 0
    variance = 0
    
    for i in range(0,len(pop)):
        x = i + popMin
        size += pop[i]
        mean += pop[i]*x
        variance += pop[i]*x*x
        
    mean /= size
    variance /= size
    variance -= mean * mean
    
    mean = round(mean,2)
    variance = round(variance, 2)
    from math import sqrt
    sd = round(sqrt(variance), 2)
        
    return [size, mean, variance, sd]