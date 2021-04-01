# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 14:14:06 2018
 
@author: jrogers1
"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import time
from random import random
from itertools import cycle, islice, combinations
from collections import defaultdict, Counter
from copy import deepcopy
default_map=np.array([[3,1,4,2,5],[3,2,3,3,2],[2,0,3,3,2],[3,0,0,3,1],[3,4,3,3,5],
        [2,3,4,4,0],[1,2,0,1,3],[0,2,0,3,2],[3,0,0,0,4],[3,1,0,4,2]]).transpose()
#transpose fixes indexing oddities 


def create_street_graph(draw=False):
    street_graph=nx.Graph()
    for i in range(6):
        for j in range(11):
            if i!=5: #add horizontal edges, if not already on the edge
                street_graph.add_edge((i,j), (i+1,j), weight=20)
            if j!=10: #add vertical edges, if not already on the edge
                street_graph.add_edge((i,j), (i,j+1), weight=15)
    street_graph.remove_edge((1,3),(2,3))
    street_graph.remove_edge((2,3),(2,4))
    street_graph.remove_edge((2,9),(3,9))
    return street_graph

def problem1():
    def get_total_distance(loc1,loc2,G): #input: 2 locations as tuples, street graph
        #e.g. get_total_distance((0,3),(4,9), create_street_graph())
        #output: distance traveled to reach all disturbances
        disturbance_map=np.array([[3,1,4,2,5],[3,2,3,3,2],[2,-1,3,3,2],[3,-1,-1,3,1],[3,4,3,3,5],
                [2,3,4,4,0],[1,2,0,1,3],[0,2,-1,3,2],[3,0,-1,0,4],[3,1,0,4,2]])
        #-1 indicates impassable terrain
        loc1_dists=nx.single_source_dijkstra(G,loc1)[0]
        loc2_dists=nx.single_source_dijkstra(G,loc2)[0]
       
        def get_node_distance(node, times_to_visit): #input: tuple (i,j), visit int
        #Use this if you are interested in navigating to the closest corner of a block.
            if times_to_visit<=0:
                return 0
            i,j=node
            node_dists=[]
            for corner in [(i,j),(i+1,j),(i,j+1),(i+1,j+1)]:
                for building in [loc1_dists, loc2_dists]:
                    node_dists.append(building[corner]) #distance to block
            return min(node_dists)*times_to_visit
           
        ans=0
        for i in range(5):
            for j in range(10):
                block=(i,j)
                times_to_visit=disturbance_map[j,i] #i broke my indexing
                ans+=get_node_distance(block,times_to_visit)
        return ans
    G=create_street_graph()
    tuples_to_check=[]
    dists=[]
    for i in range(6):
        for j in range(11):
            tuples_to_check.append((i,j))
   
    t1=time()
    for i in range(len(tuples_to_check)):
        for j in range(i+1,len(tuples_to_check)):
            b1,b2=tuples_to_check[i],tuples_to_check[j]
            d=get_total_distance(b1,b2,G)
            dists.append([d,b1,b2])
    print('done')
    dists=sorted(dists)
    print(dists[0])
    print(time()-t1, 'seconds')
    return dists[0]
 
def problem2(disturb_locs=None, number_of_facilities=2, G='default', disturbance_map=default_map):
    """
    Compute optimal placement+distance traveled for the problem.
    disturb_locs allows us to replicate a specific random scenario if desired.
    number_of_facilities allows us to place more than 2.
    G is the graph: we can add/remove connectivity restrictions.
    disturbance_map (given in transposed form) tells us where to go for disturbances.
    """
    if G=='default':
        G=create_street_graph()
    def random_disturbance_on_block():
        """
        Output: [corner_idx, mid_street_dist, reverse_dist]
        
        Selects a random place on a given block, uniformly.
        (Since horizontal streets are longer, it's a little more likely that we land on them.)
        corner_idx: [northwest, northeast, southeast, southwest]
        mid_street_dist: distance from the indexed corner to disturbance  
        reverse_dist: distance from the other corner to the disturbance
        """
        corner_idx=int(np.random.choice(4,1, p=[20/70, 15/70, 20/70, 15/70]))
        mid_street_dist=random()
        if corner_idx in [0,2]: #If we're on a horizontal street
            mid_street_dist*=20
            reverse_dist=20-mid_street_dist
        else: #If we're vertical
            mid_street_dist*=15
            reverse_dist=15-mid_street_dist
        return [corner_idx, mid_street_dist, reverse_dist]
    
    if disturb_locs==None:
        disturb_locs=defaultdict(random_disturbance_on_block)
    #disturb_locs is a dictionary of the specific locations of each incident.
    #keys: a tuple (x_coord, y_coord, incident_number).
    #values: specific random location on that block.
    
    #construct default disturbance coordinates

    default_coords=[]
    for i in range(5):
        for j in range(10):
            k=disturbance_map[i][j]
            while k>0:
                default_coords.append([i,j,k])
                k-=1
    
    
    def get_total_distance(graph, *locs, coords_of_incidents=default_coords):
        """
        Input: n locations, street graph, and optional coordinates of the form
            [[i1,j1,k1],[i2,j2,k2],...], representing which incidents we should travel to.
            if no coords: assume all disturbances on the default incident matrix.
        Output: total distance traveled + loc_incidents, a list of lists with
            the kth list containing the coordinates of the disturbances that
            the kth facility must navigate to
        """          
        loc_dists=[] #elements: dictionaries of min distance from loc to all other pts
        loc_incidents=[[] for _ in locs] #kth one is all the incidents the kth loc responds to
        for loc in locs:
            dist,path=nx.single_source_dijkstra(graph,loc)
            loc_dists.append(dist)
       
        def get_distance_for_incident(i,j,k):
            """
            Input: coordinates i,j,k, of the disturbance to check.
                k indexes the incident number.
            Output: distance.
            Effects: disturb_loc is populated with randomized disturbances,
                if not already full.
            loc_incidents is populated with coordinates, indexing which facilities
                should respond to which disturbances.
            """
            corner_idx, mid_street_distance, reverse_distance=disturb_locs[i,j,k]
            corners=cycle([(i,j),(i,j+1),(i+1,j+1),(i+1,j)])
            closest_corners=list(islice(corners, corner_idx, corner_idx+2))            
            candidate_routes=[]
            
            for n, destn in enumerate(closest_corners): 
                for loc_idx in range(len(locs)):
                    distance_to_corner=loc_dists[loc_idx][destn] #path length from a facility to destn
                    if n==0: #going to first corner, then disturbance
                        full_distance=distance_to_corner+mid_street_distance
                    if n==1: #going to second corner, then disturbance
                        full_distance=distance_to_corner+reverse_distance
                    candidate_routes.append([full_distance, loc_idx])
            
            best_dist, best_station_idx=min(candidate_routes)
            loc_incidents[best_station_idx].append((i,j,k))
            return best_dist
        

        total_distance=0        
        for i,j,k in coords_of_incidents:
            total_distance+=get_distance_for_incident(i,j,k)
        return total_distance, loc_incidents
   
    def find_best_placement(proposed_locs='all'):
        if proposed_locs=='all':
            proposed_locs=[]
            width,height=disturbance_map.shape #usually 5 by 10
            for i in range(width+1):
                for j in range(height+1):
                    proposed_locs.append((i,j))
       
        solution=[np.inf]
        comb_gen=combinations(proposed_locs,number_of_facilities)
        for x in comb_gen:
            dist,loc_incidents=get_total_distance(G,*x)
            candidate=[dist, *x, loc_incidents]
            if solution>candidate:
                solution=candidate
        return solution
   
    #loc incidents tells us where to find stuff for each ambulance etc
    best_dist, *best_locs, loc_incidents=find_best_placement()
    def tune_best_placement():
        def get_dist_from_dict(graph, n): #input: graph, #n of which loc we're looking at
            loc_n=best_locs[n]
            incidents_n=loc_incidents[n]
            return get_total_distance(graph, loc_n, coords_of_incidents=incidents_n)[0]
       
        placements=[[] for _ in range(number_of_facilities)] 
        for n in range(number_of_facilities):
            i,j=best_locs[n] #coordinates of this location
            for shift in range(0,41):
                shift/=4
                for edge in G[(i,j)].keys(): #decrease one weight, increase the other 3
                    H=deepcopy(G)
                    H[(i,j)][edge]['weight']-=shift
                    for other_edge in H[(i,j)].keys():
                        if other_edge!=edge:
                            H[(i,j)][other_edge]['weight']+=shift
                    dist=get_dist_from_dict(H,n)
                    a,b=edge
                    where=[a-i, b-j]
                    if where==[0,1]:
                        d='South'
                    elif where==[0,-1]:
                        d='North'
                    elif where==[1,0]:
                        d='East'
                    elif where==[-1,0]:
                        d='West'
                    elif where==[0,0]:
                        d='Corner'
                    else:
                        raise
                        
                    placements[n].append([dist,shift,i,j,d])
        best_spots=[min(x) for x in placements]
        shifts=[x[1] for x in best_spots]
        dirs=[x[4] for x in best_spots]
        return dirs, shifts
    wiggled=tune_best_placement()
    print(wiggled)

    return [[best_dist], best_locs, wiggled[0], wiggled[1]]

#please_work=problem2(number_of_facilities=2)
#defaults=please_work[-1]



def simulate_and_count(n=3):
    answers=[]
    ind=Counter()
    pairs=Counter()
    shifts=Counter()
    dirs=Counter()
    for _ in range(n):
        ans=problem2()
        answers.append(ans)
        best_locations=tuple(ans[1])
        ind.update(ans[1])
        shifts.update(ans[-1])
        pairs[best_locations]+=1
        dirs.update(ans[2])
    print('individual locations: {}'.format(ind))
    print('pairs: {}'.format(pairs))
    print('shifts: {}'.format(shifts))
    print('dirs: {}'.format(shifts))
    return ind, pairs, shifts, ans
simulate_and_count(n=100)