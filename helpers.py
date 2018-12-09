from graphviz import Digraph

def draw_samecoin_graph(g, highlight_nodes=[], color='yellow'):  
    
    g.attr(rankdir='LR', ranksep='0.5')
    g.attr('node', shape='circle', fontsize='10')
    g.attr('edge', fontsize='10')

    g.node('Root','R')
    g.node('F') # Fair coin
    g.node('L') # loaded coin
    
#   print('noding')
    n_flips = 3  # (F or L is not considered a flip)
    i_outcome = 1
    for each_flip in range(1,n_flips+1):
        n_outcomes = 2**each_flip
        for each_outcome in range(0, n_outcomes):            
            new_H = 'H{}'.format(i_outcome) 
            new_T = 'T{}'.format(i_outcome)             
            g.node(new_H, 'H')
            g.node(new_T, 'T')                        
            i_outcome += 1
        
    # choose F or L 
    g.edge('Root','F',label='0.5')
    g.edge('Root','L',label='0.5')
    
    # flip 1 of H/T (F or L is not considered a flip)
    g.edge('F','H1',label='0.5')
    g.edge('F','T1',label='0.5')
    g.edge('L','H2',label='0.9')
    g.edge('L','T2',label='0.1')
    
    # flip 2 of H/T - Fair coin
    g.edge('H1','H3',label='0.5')
    g.edge('H1','T3',label='0.5')
    g.edge('T1','H4',label='0.5')
    g.edge('T1','T4',label='0.5')
    
    # flip 2 of H/T - Loaded coin
    g.edge('H2','H5',label='0.9')
    g.edge('H2','T5',label='0.1')
    g.edge('T2','H6',label='0.9')
    g.edge('T2','T6',label='0.1')   
    
    # flip 3 of H/T - Fair coin
    g.edge('H3','H7',label='0.5')
    g.edge('H3','T7',label='0.5')
    g.edge('T3','H8',label='0.5')
    g.edge('T3','T8',label='0.5')
    g.edge('H4','H9',label='0.5')
    g.edge('H4','T9',label='0.5')
    g.edge('T4','H10',label='0.5')
    g.edge('T4','T10',label='0.5')
    
    # flip 4 of H/T - Loaded coin
    g.edge('H5','H11',label='0.9')
    g.edge('H5','T11',label='0.1')
    g.edge('T5','H12',label='0.9')
    g.edge('T5','T12',label='0.1')
    g.edge('H6','H13',label='0.9')
    g.edge('H6','T13',label='0.1')
    g.edge('T6','H14',label='0.9')
    g.edge('T6','T14',label='0.1')       
    
    
    for each_node in highlight_nodes:
        #print(each_node)
        g.node(each_node,style='filled',fillcolor=color)

    return g

def update_samecoin_graph(g, highlight_nodes=[], color='yellow'):
    for each_node in highlight_nodes:
        #print(each_node)
        g.node(each_node,style='filled',fillcolor=color)
    return g


