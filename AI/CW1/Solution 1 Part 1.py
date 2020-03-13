#!/usr/bin/env python
# coding: utf-8

# In[2]:


import json
import math
from graphviz import Graph
import node as Node

# load the data
with open('UK_cities.json') as f:
    data = json.load(f)    
# print(data)

#visualise the data
u = Graph('cities', filename='cities.gv',
            node_attr={'color': 'lightblue2', 'style': 'filled'})
u.attr(size='6,6')

for k0,k1 in data.items():
    for k,v in k1.items():
        u.edge(k0,k,label=str(v['weight']))    
u.view()


# In[3]:


def is_environment_aware(environment_aware_factor,distance,speed):
    '''
    if the environment_aware_factor is 0, return 
        cost as the distance
        if environment_aware_factor is 1 calculat-
        e the cost as the sum of the overall driv-
        ing time plus the overall cost of the air 
        pollution due to your driving
        if environment_aware_factor is -1 calculat-
        e the cost as overall cost is the sum of t-
        he car rental fee plus the total (likely) 
        fines
    params, environment_aware_factor, integer, (0,
        1,-1)
    return, cost
    '''
    if environment_aware_factor == 0:
        cost = distance
    elif environment_aware_factor == 1:
        cost = (distance/speed)+(0.00001*speed*speed)*(distance/speed)
    elif environment_aware_factor == -1:
        speed_limit = distance # assumption as mentioned in question
        # fine_factor is a value between 0 and 1
        fine_factor = 0 if speed < speed_limit else 1 - math.exp(-(speed-speed_limit))
        # if fine_factor > 0 put 1000 fine, else 0
        fine =  1000 * (1 if fine_factor > 0 else 0)
        car_rental = (distance/speed)*100
        cost = car_rental + fine
    else:
        cost = 0
    return cost


# In[4]:


# preprocessing step 1
city_list = []
city_weight_map_list = []
environment_aware = 0
# creating a map of all nodes to all nodes where path exists.

for k0,k1 in data.items():
    for k,v in k1.items():
        city_list.append(k0)
        city_list.append(k)
        city_weight_map_list.append([k0,k,is_environment_aware(-1,float(v['weight']),316.22)])

city_list = list(set(city_list))


# In[5]:


# preprocessing step 2
# generating a dict of the above map
city_map_dict = {}
for i in city_list:
    alist = []
    for j in city_weight_map_list:
        if i in j:
            alist.append(j)
            city_map_dict[i] = alist


# In[6]:


print(city_map_dict['london'])


# In[8]:


graph = {}
# generate a graph of the above map 
for k,v in city_map_dict.items():
    _reachable = []
    for i in v:
        _reachable.append(Node.Node(i[0] if i[0] != k else i[1], 0, i[2], None))
    graph[k] = _reachable
    


# In[9]:


graph['london'][0].return_state()


# In[10]:


def return_node_cost(node):
    return node.return_cost


# In[11]:


import gc

def ucf(graph,start,destination):
    
    def recursive_expand_nodes(graph,frontier,visited_nodes,destination,previous_frontier):
        if len(frontier) is not 0: # if empty return failure
            node = frontier[0] # assign and pop the lowest cost node from front
            frontier.remove(node)
            if node.return_state() not in visited_nodes: # if the current node is not visited then
                previous_frontier = [node.return_state()] if previous_frontier is None else visited_nodes
                print("frontier = ",[(node.return_state(),str(node.return_cost()),previous_frontier)])
                child_node_list = graph[node.return_state()] # get all the possible path from the current node as children
                # if current_node is not equal to destination node, expand its children
                if node.return_state() not in [destination]: 
                    # for each child in parent_node set cost, depth and add the child to frontier for it to be explored 
                    print('Frontier selected & Child nodes of',node.return_state(),'are')
                    child_list = []
                    for child_node in child_node_list:                        
                        child_node.put_cost(node.return_cost() + child_node.return_cost())
                        child_node.put_depth(node.return_depth()+1)
#                         print("increasing depth to",child_node.return_depth(), child_node.return_state())
                        child_list.append(['depth '+str(child_node.return_depth()),child_node.return_state(),'total cost incured',child_node.return_cost()])
                        child_node.put_node(node)
                        frontier.append(child_node)
                    v = [(i.return_state(),i.return_depth(),i.return_cost()) for i in frontier]
                    print("Exploring all frontiers = ")
                    for i in v:
                        print(str([(node.return_state(),str(node.return_cost())),i]))
                    frontier.sort(key=lambda nodes: nodes.cost) # sort frontier according to lowest cost
                    print("Sorting frontier based on cost  = ",[(node.return_state(),str(node.return_cost()),[(i.return_state(),i.return_depth(),i.return_cost()) for i in frontier])])
                    
                visited_nodes.append(node.return_state()) # add the current node to visited node
                #print(frontier)
                print("explored =",set(visited_nodes))
                print("destination =",destination)
                print("*"*25)
                
                # if current node is destination node then return with the path and cost
                if node.return_state() in [destination]:
                    previous_nodes = []
                    previous_nodes.append(node)
                    while node.return_node() is not None:
                        node = node.return_node()
                        previous_nodes.append(node)

                    previous_nodes = previous_nodes[::-1] # reversing the order

                    count = 0
                    print("Path =")
                    for child in previous_nodes:                        
                        if len(previous_nodes)-1 > count:
                            count = count + 1
                            print(child.return_state(),"->",previous_nodes[count].return_state(),"," ,str(previous_nodes[count].return_cost() - child.return_cost())," units")

                        # if london to london
                        if len(previous_nodes) == 1:
                            print(child.return_state(),"->",child.return_state(),",",str(previous_nodes[-1].return_cost()),"units")
                    print("Total Cost =",str(previous_nodes[-1].return_cost())," units")
                    del frontier    
                    return "Completed trace"               
            recursive_expand_nodes(graph,frontier,visited_nodes,destination,previous_frontier)
        else:
            print("Distance: Infinity")
            print("Route: None")
        
    frontier = []
    visited_nodes = []
    frontier.append(Node.Node(start))
    recursive_expand_nodes(graph,frontier,visited_nodes,destination,None)

    gc.collect()


# In[12]:


ucf(graph,'london','aberdeen')


# In[ ]:





# In[ ]:




