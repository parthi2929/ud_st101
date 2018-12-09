from graphviz import Digraph
import colorsys
import random


def HSVToHex(h, s, v):
    (r, g, b) = colorsys.hsv_to_rgb(h, s, v)
    hexy = "".join("%02X" % round(r*255)) + "".join("%02X" % round(g*255)) + "".join("%02X" % round(b*255))
    return hexy
 
def getDistinctColors(n):
    huePartition = 1.0 / (n + 1)
    saturation = 0.2
    lightness = 0.9
    colors_list = [HSVToHex(huePartition * value, saturation, lightness) for value in range(0, n)]
    return colors_list#, colors_comp_list

def draw_graph(g, n_outcomes = 2, n_flips=2):
    """
    Given no of flips, this function creates a corresponding probability tree
    For now, assuming outcomes are numbered
    """
    #g.attr(rankdir='LR', ranksep='0.5')
    g.attr('node', shape='circle', fontsize='10')
    g.attr('edge', fontsize='10')
    g.node('Root','R',style='filled', fillcolor='#DCDCDC')    # first node
    
    i_kids = 1
    parent_list = []
    
    # colors for each outcome
    colors_list = getDistinctColors(n_outcomes)
    #print(colors_list)
    
    # for each flip
    for each_flip in range(1, n_flips+1):
        per_flip_outcomes = n_outcomes**(each_flip)
        
        # for each node outcome of flip
        temp_list = []
        p_index = 0 # parent index for each node
        for each_outcome in range(0, int(per_flip_outcomes/n_outcomes)):
            
            
            # draw nodes, record parents
            for each_kid in range(0, n_outcomes):                
                
                new_kid_ID = '{}'.format(i_kids) # Id the kid
                new_kid_label = str(each_kid)
                g.node(new_kid_ID, new_kid_label, style='filled', fillcolor='#{}'.format(colors_list[each_kid]), fontcolor='black')
                               
                parents = parent_list[-1] if len(parent_list) > 0 else []
                parent = parents[p_index] if len(parents) > 0 else None
                
                #debug
                #print('Flip:{} Outcome:{} Kid\'s Label:{} Possible ID:{} Parent:{}'.format(each_flip, each_outcome, each_kid, i_kids, parent))    
                
                i_kids += 1
                
                # draw edges
                if parent is not None:
                    g.edge(parent, new_kid_ID)
                else: 
                    g.edge('Root', new_kid_ID)
                
                # for next set of kids                    
                temp_list.append(new_kid_ID)
                
            p_index += 1

        parent_list.append(temp_list)
        print()
                
    return g

import pandas as pd 

def sum_digits(n):
    n = int(n)
    s = 0
    while n:
        s += n % 10
        n //= 10
    return s

def get_combinations(n_outcomes = 2, n_flips=2):
    """
    Given the no of flips, this function will provide the final sequence combinations as a panda dataframe
    """
    # setup data frame with necessary cols
    columns = ['sequence', 'x']
    df = pd.DataFrame(columns=columns)
    
    # generate the individual outcomes
    outcomes = list(range(0,n_outcomes))  # so if 3, its 0,1,2
    outcomes = [str(i) for i in outcomes]  # convert all to string
    
    # get the combinations
    from itertools import product
    for i in product(outcomes, repeat=n_flips):  
        combi = ''.join(i)
        summy = sum([int(x) for x in i])
        #print(i,combi, summy)
        df = df.append({'sequence': combi, 'x': summy }, ignore_index=True)
        
    # get no of heads in the combinations
    #print('Given no of flips:', n_flips)
    #print('\nx = no of heads in respective sequence')
        
    return df

def get_combinations_consolidated(n_outcomes = 2, n_flips=2):
    """
    Given the raw dataframe of combinations, this will provide n(x) and p(x)
    """
    # setup data frame with necessary cols
    columns = ['x', 'n(x)', 'p(x)']
    df = pd.DataFrame(columns=columns)
    
    # get raw data
    combi_df = get_combinations(n_outcomes=n_outcomes, n_flips=n_flips)
    x_list = combi_df['x'].tolist()
    
    # extract frequency
    #ref: https://stackoverflow.com/questions/2161752/how-to-count-the-frequency-of-the-elements-in-a-list/2162045
    x_list.sort()
    from itertools import groupby 
    freq_tuple = [ (key, len(list(group))) for key, group in groupby(x_list)]
    #print(freq_tuple)
    
    for each_freq_tuple in freq_tuple:
        x = each_freq_tuple[0]
        n_x = each_freq_tuple[1]
        p_x = n_x/(n_outcomes**n_flips)  # its a conditional probability, thats y divided by total outcomes
        df = df.append({'x': x, 'n(x)': n_x, 'p(x)': p_x }, ignore_index=True)
        
    # convert cols to integer (except p(x))
    df[['x','n(x)']] = df[['x','n(x)']].astype(int) #ref: https://stackoverflow.com/questions/21291259/convert-floats-to-ints-in-pandas/21291622

    #print('n(x) = total no of possible x type sequences')
    #print('for eg, if x = 2, n(x) = 3, then there are 3 possible sequence types, in each of which, no of heads is 2')    
    
    #print('\np(x) = conditional probability that n(x) could occur out of all outcomes')
    
    return df

import matplotlib.pyplot as plt

def autoformat(ax, xlabel, ylabel, fontsize):
    """
    Few tweaks for better graph
    """
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(fontsize) # Size here overrides font_prop
        
    ymin = ax.get_ylim()[0]
    ymax = ax.get_ylim()[1]*1.1  # increase space to insert bar value
    ax.set_ylim([ymin,ymax])    

    # x values should be integers as its no of heads
    xmin = ax.get_xlim()[0]-1
    xmax = ax.get_xlim()[1]
    from math import ceil,floor
    xminint = floor(xmin)+1
    xmaxint = ceil(xmax)+1
    xint = range(xminint, xmaxint)
    ax.set_xticks(xint)


def autolabel(ax, rects, fontsize):
    """
    Attach a text label above each bar displaying its height
    ref: https://matplotlib.org/2.0.2/examples/api/barchart_demo.html
    """    
    for rect in rects:
        height = rect.get_height()
        #text = '%.4f' % height
        text = '{0: <{width}}'.format(height, width=1) 
        ax.text(rect.get_x() + rect.get_width()/2., 1.005*height,text, ha='center', va='bottom', fontsize=fontsize+3, color='red')

def plot_combinations_consolidated(df, fontsize=10, label=True):
    """
    Given the dataframe with x, n(x), p(x) this provides two plots:
    x vs n(x)
    x vs p(x)
    """
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))
    
    X = df['x'].tolist()
    N = df['n(x)'].tolist()
    P = df['p(x)'].tolist()
    
    rects = ax1.bar(X, N)
    if label == True:
        autolabel(ax1, rects, fontsize)
    
    xlabel = 'Sum of each outcome in a Sequence'
    ylabel = 'No of Sequences\nhaving that Sum'   
    autoformat(ax1, xlabel, ylabel, fontsize)
    

    rects = ax2.bar(X, P)
    if label == True:
        autolabel(ax2, rects, fontsize)
    
    ylabel = 'Probability of ANY of Sequences\nhaving that Sum'   
    autoformat(ax2, xlabel, ylabel, fontsize)
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.45)    
    plt.show()



def ncr(n, r):
    """
    Calculate n choose r
    https://stackoverflow.com/questions/4941753/is-there-a-math-ncr-function-in-python
    """
    import operator as op
    from functools import reduce
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer//denom

def subsets_turbo(n,p,s):
    """
    Finding no of combinations whose sum is p, for s-sided dice for n-no of flips (or n dices) using formula
    http://mathworld.wolfram.com/Dice.html
    """
    from math import floor
    k_max = floor((p-n)/s)
    
    # summation
    result = 0
    for i in range(0, k_max+1):  # k_max inclusive..
        
        C_1 = (-1)**i
        C_2 = ncr(n, i)
        C_3 = ncr(p-s*i-1,n-1)
        C_4 = C_1*C_2*C_3
        result += C_4
        
    return result



def get_combinations_consolidated_turbo(n_outcomes = 3, n_flips = 3):
    """
    Given the  no of outcomes per flip, no of flips, and probability of each sum 
    this will provide dataframe: x, n(x) and p(x)
    WITHOUT CALCULATING INDIVIDUAL COMBINATIONS
    Formula: http://mathworld.wolfram.com/Dice.html
    """
    n = n_flips # no of flips
    
    s = range(1,n_outcomes+1) # s-sided = 1,2,3
    
    n_s = len(s)
    p_min = min(s)*n
    p_max = max(s)*n
    p = range(p_min, p_max+1)  # 2,3,4,5,6, p is not probability, but sum, letter p indicating 'points' of dice   
    
    try:
        X = p
        N = [subsets_turbo(n,i,n_s) for i in p]
        df = pd.DataFrame({'x': X, 'n(x)': N}, dtype='object') #dtype needed to avoid pandas overflow error
        total_outcomes = df['n(x)'].sum()
        df['p(x)'] = df['n(x)']/total_outcomes
        df = df[['x','n(x)','p(x)']]
        return df
    except Exception as e:
        e = str(e)
        print("X:{} Max(N):{} len(N):{} Unexpected error:{}".format(X, max(N), len(N), e))
        raise

# statistical toss functions.. 
from random import choice, seed
import numpy as np
seed(0)  # just for consistent result every time..   
def toss(n_toss, n_outcomes):
    """
    Toss given dice with n_outcomes, for n_toss times. Each outcome has equal probability.
    Thus a single toss gives a uniform distribution. 
    """

    final_sequence = []
    
    s = range(1,n_outcomes+1) # s-sided = 1,2,3    
    n_s = len(s)
    
    for i in range(n_toss):  # 0 to (n_toss-1) times..
        toss_result = np.random.choice(s) # uniform distribution assumed
        final_sequence.append(toss_result)
        #rint(toss_result)
    return sum(final_sequence)

def sample(n_experiments, n_toss, n_outcomes):
    """
    Conduct experiment given no of times
    In each experiemnt, toss given no of times, and update n_X
    """
    from collections import defaultdict
    samples = defaultdict(lambda: 0)
    for each_experiment in range(0, n_experiments):
        X = toss(n_toss, n_outcomes)  # X is sum of outcome sequence of n_toss        
        samples[X] += 1   # constructing n(X)
        #print(each_experiment, dict(samples))
        
    # convert to pandas
    df = pd.DataFrame([[key,value] for key,value in samples.items()],columns=['x','n(x)'])
    df.sort_values('x', inplace=True)
    total_outcomes = df['n(x)'].sum()
    df['p(x)'] = df['n(x)']/total_outcomes
    return df