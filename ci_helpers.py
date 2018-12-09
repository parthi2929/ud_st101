from random import shuffle 
import matplotlib.pyplot as plt
from SDSPSM import get_metrics
from random import choices, seed
from math import sqrt, pi

def create_bernoulli_population(T, p):
    y_freq = int(p*T)                 
    y_pops = [1]*y_freq        
    o_freq = int((1-p)*T)
    o_pops = [0]*o_freq
    population = y_pops + o_pops
    population_freq = [o_freq, y_freq]
    shuffle(population) 
    return population, population_freq

from random import random
from math import floor
def createRandomPopulation(N, freqMax):
    """
    Create a random distribution for N values. This is to illustrate Sampling Distribution of Sample Means. 
    """
    population = []
    population_freq = []
    for i in range(0,N):
        temp_freq = (floor(random() * freqMax))  # random frequency for each population
        temp_list = [i]*temp_freq
        population += temp_list
        population_freq.append(temp_freq)
    shuffle(population)
    return population, population_freq


def mini_plot_SDSP(raw_list, ax1,ax2,ax3, norm_off=False, width=0.1): 
    ax1.hist(raw_list) 
    ax1.set_title('Distribution')
    
    _, bins,_ = ax2.hist(raw_list, density=True) 
    ax2.set_title('Density')
    
    # probability mass
    dummy_dict = {i:raw_list.count(i) for i in raw_list}
    total = sum(list(dummy_dict.values()))
    pmf = {key: round(val/total,4) for key, val in dummy_dict.items()}
    ax3.bar(list(pmf.keys()), list(pmf.values()), width=width)
    ax3.set_title('PMF')
    
    mu, var, sigma = get_metrics(raw_list)
    metrics_text = '$\mu_x:{}$ \n$\sigma_x:{}$'.format(mu, sigma)
    ax2.text(0.97, 0.98,metrics_text,ha='right', va='top',transform = ax2.transAxes,fontsize=10,color='red')   

    # normal approx overlay if needed
    if norm_off == False:  # so user wants normal curve overlay
        
        import numpy as np
        X = np.linspace(min(bins),max(bins),10*len(bins))
        from math import sqrt, pi
        Cp = 1/(sigma*sqrt(2*pi))
        Ep = -1/2*((X-mu)/sigma)**2
        G = Cp*np.exp(Ep)      
        ax2.plot(X, G, color='red')


def mini_plot_SDSM(raw_list,ax1,ax2,ax3, bins, width=0.5): 

    ax1.hist(raw_list, bins) 
    ax1.set_title('Distribution')
    
    n, bins,_ = ax2.hist(raw_list, bins, density=True) 
    ax2.set_title('Density')

    # probability mass
    dummy_dict = {i:raw_list.count(i) for i in raw_list}
    total = sum(list(dummy_dict.values()))
    pmf = {key: round(val/total,4) for key, val in dummy_dict.items()}
    ax3.bar(list(pmf.keys()), list(pmf.values()), width=width)
    ax3.set_title('PMF')

def sample_with_CI(N, n, population, sigma=1, mode='z'):
    """
    N - no of trials/experiments
    n - sample size
    sigma - population SD (needed in z mode)
    """
    Y_hat = []
    Y_mean_list = []
    CI_list = []
    for each_experiment in range(N):  

        Y_hat = choices(population, k=n)   # sample with replacement
        Y_mean = sum(Y_hat)/len(Y_hat)
        
        if mode == 'z': # then use population SD sigma, not typically practical 
            c = 1.96  # which comprises of 95% of data points in std. normal distribution
            Y_sigma = sigma
            CI_err = round(c*(Y_sigma/sqrt(n)),4)
        else:  # t distribution, use unbiased variance
            from scipy import stats
            c = stats.t.ppf(1-0.025, n-1)  # t value varies depending on degrees of freedom
            Y_variance = sum([(y - Y_mean)**2 for y in Y_hat])/(n-1)  # unbiased estimator
            Y_sigma = round(sqrt(Y_variance), 4)
            CI_err = round(c*(Y_sigma/sqrt(n)),4)
        
        CI_list.append((Y_mean, CI_err))
        Y_mean_list.append(Y_mean)
        
    return Y_mean_list, CI_list

def plot_ci_accuracy_1(ax, CI_list, mu):
    
    mean, err = zip(*CI_list)
    index = range(0,len(mean))
    err_count = 0
    
    # each CI interval, check if it contains mu or not
    for each_ci in index:
        each_mid = mean[each_ci]
        each_err = err[each_ci]
        low_err = each_mid - each_err
        hig_err = each_mid + each_err
        c = 'C0'
        if (hig_err <= mu) or (low_err >= mu): # outliers
            c = 'C1'
            err_count += 1
        ax.errorbar(each_ci,each_mid, yerr=each_err, fmt='o', color=c)
    # cosmetics    
    ax.axhline(y=mu, color='r')
    ax.set_xticks(index)
    ax.xaxis.grid(True, alpha=0.3)
    accuracy = (1-round(err_count/len(mean),4))*100
    print('CI containing pop.mean:{}%'.format(accuracy))
    
    return accuracy

def get_ci_accuracy(CI_list, mu):
    mean, err = zip(*CI_list)
    index = range(0,len(mean))
    err_count = 0
    
    # each CI interval, check if it contains mu or not
    for each_ci in index:
        each_mid = mean[each_ci]
        each_err = err[each_ci]
        low_err = each_mid - each_err
        hig_err = each_mid + each_err
        if (hig_err <= mu) or (low_err >= mu): # outliers
            err_count += 1

    accuracy = (1-round(err_count/len(mean),4))*100 
    return accuracy

from random import sample
def repeated_experiments_with_CI(population,mu, sigma, N_list=[], n_list=[],  mode=1, dist=0, format='b'):
    """
    population - raw population from which sample to be taken
    mu, sigma - population parameters
    C - 85% CI constant (for eg, 1.96 or 2.093 etc)
    N_list - Experiment size or no of times experiments to be conducted per trial
    n_list - Sample size or no of samples per experiment
    Mode 1 - use population SD
    Mode 2 - use unbiased sample SD
    Mode 3 - usebiased sample SD
    dist - 0 - use Z distribution
    dist - 1 - use t distribution
    """
    accuracy_list = []
    for each_N in N_list:  # no of experiments
        for each_n in n_list:   # sample size for each experiment
            err_count = 0
            for each_E in range(each_N): # for each experiment of experiment size
                Y_hat = sample(population, k=each_n)  # pick n samples
                Y_mean = sum(Y_hat)/len(Y_hat)
                
                if dist == 0:
                    C = 1.96
                elif dist == 1:
                    from scipy import stats
                    C = stats.t.ppf(1-0.025, each_n-1)  # t value varies depending on degrees of freedom  which is (no of samples - 1)             

                if mode == 1:
                    Y_sigma = sigma
                    CI_err = round(C*(Y_sigma/sqrt(each_n)),4)
                elif mode == 2:
                    Y_variance = sum([(y - Y_mean)**2 for y in Y_hat])/(each_n-1)  # unbiased estimator
                    Y_sigma = round(sqrt(Y_variance), 4)
                    CI_err = round(C*(Y_sigma/sqrt(each_n)),4)
                elif mode == 3:
                    Y_variance = sum([(y - Y_mean)**2 for y in Y_hat])/(each_n)  # biased estimator
                    Y_sigma = round(sqrt(Y_variance), 4)
                    CI_err = round(C*(Y_sigma/sqrt(each_n)),4)
                else:
                    raise ValueError('Wrong mode')
                low_err = Y_mean - CI_err
                hig_err = Y_mean + CI_err
                #print(CI_err,Y_mean,low_err,hig_err,mu)
                if (hig_err <= mu) or (low_err >= mu):
                    err_count += 1
            accuracy = round((1-err_count/each_N)*100,4)
            success = 0 if accuracy < 95 else 1
            if format == 'b':
                accuracy_list.append((each_N, each_n, success))
            else: # anything other than b
                accuracy_list.append((each_N, each_n, accuracy))
    return accuracy_list

def get_mode_label(mode):
    mode_labels={
        1:"population SD",
        2:"unbiased sample SD",
        3:"biased sample SD",
        4:"Wilson Score method"
    }
    return mode_labels.get(mode, "invalid mode")

def get_dist_label(dist):
    dist_labels={
        0:"Z distribution",
        1:"T distribution"
    }
    return dist_labels.get(dist, "invalid dist")

def plot_ci_accuracy_2(ax, accuracy_list):
    x,y,z=zip(*accuracy_list) 
    labels = ['$< 95\%$', '$ \geq 95\%$']
    colors = ['red','green']
    for xi, yi, zi in zip(x,y,z):
        if zi == 1:  
            s = ax.scatter(yi, xi, c=colors[zi], label=labels[zi], s=30, edgecolors='None', alpha=0.75)
        else:
            f = ax.scatter(yi, xi, c=colors[zi], label=labels[zi], s=30, edgecolors='None', alpha=0.75)

    ax.set_ylabel('Experiment Size $N$')
    ax.set_xlabel('Sample Size $n$')
    #ax.legend((s,f),(labels[1],labels[0]), loc='lower right', scatterpoints=1)

    xmin = 10  # 0 does not make sense..
    xmax = ax.get_xlim()[1]
    from math import ceil,floor
    xminint = floor(xmin)
    xmaxint = ceil(xmax)
    xint = range(xminint, xmaxint, 50)
    ax.set_xticks(xint)
    ax.xaxis.set_tick_params(labelsize=7)
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)

def repeated_experiments_with_CI_madmax(population,mu, sigma, each_N, each_n,  mode=1, dist=0, format='b', replace=True):
    """
    population - raw population from which sample to be taken
    mu, sigma - population parameters
    C - 85% CI constant (for eg, 1.96 or 2.093 etc)
    N_list - Experiment size or no of times experiments to be conducted per trial
    n_list - Sample size or no of samples per experiment
    Mode 1 - use population SD
    Mode 2 - use unbiased sample SD
    Mode 3 - usebiased sample SD
    dist - 0 - use Z distribution
    dist - 1 - use t distribution
    """
    accuracy_list = []
    err_count = 0
    T = len(population)
    for each_E in range(each_N): # for each experiment of experiment size

        if replace == True:
            Y_hat = choices(population, k=each_n)  # choices by default samples with replacement
            fpc = 1
        else:
            Y_hat = sample(population, k=each_n)   # sample by default samples without replacement
            fpc = sqrt((T-each_n)/(T-1))
        Y_mean = sum(Y_hat)/len(Y_hat)

        if dist == 0:
            C = 1.96
        elif dist == 1:
            from scipy import stats
            C = stats.t.ppf(1-0.025, each_n-1)  # t value varies depending on degrees of freedom  which is (no of samples - 1)             

        if mode == 1:
            Y_sigma = sigma*fpc          # fpc affects depending on with or without replacement
            CI_err = round(C*(Y_sigma/sqrt(each_n)),4)
        elif mode == 2:
            Y_variance = sum([(y - Y_mean)**2 for y in Y_hat])/(each_n-1)  # unbiased estimator
            Y_sigma = round(sqrt(Y_variance), 4)
            Y_sigma = Y_sigma*fpc
            CI_err = round(C*(Y_sigma/sqrt(each_n)),4)
        elif mode == 3:
            Y_variance = sum([(y - Y_mean)**2 for y in Y_hat])/(each_n)  # biased estimator
            Y_sigma = round(sqrt(Y_variance), 4)
            Y_sigma = Y_sigma*fpc
            CI_err = round(C*(Y_sigma/sqrt(each_n)),4)
        else:
            raise ValueError('Wrong mode')
        low_err = Y_mean - CI_err
        hig_err = Y_mean + CI_err
        #print(CI_err,Y_mean,low_err,hig_err,mu)
        if (hig_err <= mu) or (low_err >= mu):
            err_count += 1
    accuracy = round((1-err_count/each_N)*100,4)
    success = 0 if accuracy < 95 else 1
    if format == 'b':
        accuracy_list.append((each_N, each_n, success))
    else: # anything other than b
        accuracy_list.append((each_N, each_n, accuracy))
    return accuracy_list

def plot_ci_accuracy_3(ax, accuracy_list, vmin=60, vmax=100):
    # for continous format of accuracy list.. gradient coloring
    y,x,z=zip(*accuracy_list) 
    

    points = ax.scatter(x, y, c=z, cmap='PiYG', vmin=vmin, vmax=vmax)
    
    
    xmin = 10  # starting with sample size 10
    xmax = ax.get_xlim()[1]
    from math import ceil,floor
    xminint = floor(xmin)
    xmaxint = ceil(xmax)
    xint = range(xminint, xmaxint, 50)
    ax.set_xticks(xint)
    ax.xaxis.set_tick_params(labelsize=7)
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
        
    ax.set_ylabel('Experiment Size $N$')
    ax.set_xlabel('Sample Size $n$')
        
    return points

def plot_ci_accuracy_3_1(ax, accuracy_list, fig):
    lst = [x[2] for x in accuracy_list]
    if (len(set(lst)) <= 1):  # all elements are same, so no point in color gradient
        y,x,z=zip(*accuracy_list) 
        points = ax.scatter(x, y, c='g')
        import matplotlib.patches as mpatches
        recs = []
        recs.append(mpatches.Rectangle((0,0),1,1,fc='g'))
        ax.legend(recs,[lst[0]])#,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    else:
        points = plot_ci_accuracy_3(ax, accuracy_list, vmin=90, vmax=100)
        import numpy as np
        fig.colorbar(points, ax=ax, extend='min' )#, ticks=list(range(vmin,vmax+vstep,vstep)))

def plot_summary(population, N_list, n_list):
    fig, axarr = plt.subplots(4,2, figsize=(15,20))  
    m , _ , s = get_metrics(population)
    accuracy_list_z_wor = []
    accuracy_list_t_wor = []
    accuracy_list_z_wr = []
    accuracy_list_t_wr = []   

    for i in range(1,3): # 1,2 modes (biased SD mode ignored for now)

        for each_N in N_list:
            for each_n in n_list:

                dist=0  # z dist
                ax = axarr[0,i-1]
                accuracy = repeated_experiments_with_CI_madmax(population,m,s,each_N,each_n,i,dist,'f', replace=False)
                accuracy_list_z_wor.append(accuracy[0])
                ax.set_title('Fig 0{}: Using {} and {}'.format(i,get_dist_label(dist), get_mode_label(i)))

                dist=1   # t dist
                ax = axarr[1,i-1]
                accuracy = repeated_experiments_with_CI_madmax(population,m,s,each_N,each_n,i,dist,'f', replace=False)
                accuracy_list_t_wor.append(accuracy[0])
                ax.set_title('Fig 1{}: Using {} and {}'.format(i,get_dist_label(dist), get_mode_label(i)))

                dist=0  # z dist
                ax = axarr[2,i-1]
                accuracy = repeated_experiments_with_CI_madmax(population,m,s,each_N,each_n,i,dist,'f', replace=True)
                accuracy_list_z_wr.append(accuracy[0])
                ax.set_title('Fig 2{}: Using {} and {}'.format(i,get_dist_label(dist), get_mode_label(i)))

                dist=1   # t dist
                ax = axarr[3,i-1]
                accuracy = repeated_experiments_with_CI_madmax(population,m,s,each_N,each_n,i,dist,'f', replace=True)
                accuracy_list_t_wr.append(accuracy[0])
                ax.set_title('Fig 3{}: Using {} and {}'.format(i,get_dist_label(dist), get_mode_label(i)))            


        ax = axarr[0,i-1]
        plot_ci_accuracy_3_1(ax, accuracy_list_z_wor, fig)

        ax = axarr[1,i-1]
        plot_ci_accuracy_3_1(ax, accuracy_list_t_wor, fig)

        ax = axarr[2,i-1]
        plot_ci_accuracy_3_1(ax, accuracy_list_z_wr, fig)

        ax = axarr[3,i-1]
        plot_ci_accuracy_3_1(ax, accuracy_list_t_wr, fig)


    plt.subplots_adjust(hspace=0.5)    
    plt.figtext(0.5,0.9,'Sampling Without Replacement', ha='center', va='center', fontsize=16, color='b')
    plt.figtext(0.5,0.49,'Sampling With Replacement', ha='center', va='center', fontsize=16, color='b')
    # plt.tight_layout()
    plt.show()

#sdg