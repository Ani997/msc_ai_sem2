#!/usr/bin/env python
# coding: utf-8

# In[1]:


from amnesiac import blurry_memory
import random
import numpy as np
import matplotlib.pyplot as plt
import time
import string


# In[2]:


random.seed(10)


# In[3]:


def fitness(population,user_id,passwd_no):
    return [[i,int(blurry_memory([''.join(i)],user_id,passwd_no)[''.join(i)]*10)] for i in population]


# In[4]:


def select_parents(fitness_scores,method='rank'):
    if method == 'rank':
        return [i[0] for i in sorted(fitness_scores, key=lambda x: x[1], reverse = True)[:num_parents]]
    if method == 'mixed_rank':
        fitness_scores_sorted = sorted(fitness_scores, key=lambda x: x[1], reverse = True)
        parents = [i[0] for i in fitness_scores_sorted]
        selected_parents = parents[:int(num_parents*0.6)]+parents[-int(num_parents*0.2):]
        return random.choices(parents[int(num_parents*0.6):-int(num_parents*0.2):],k=int(num_parents*0.2)) + selected_parents
    if method == 'tournament':
        parents_list = []
        while len(parents_list) <= num_parents:
            best_random_tounamenet_players = random.choices(fitness_scores,k=3)
            best_random_tounamenet_players = sorted(best_random_tounamenet_players, key=lambda x: x[1], reverse = True)
            parents_list.append(best_random_tounamenet_players[0][0])
            fitness_scores.remove(best_random_tounamenet_players[0])
        return parents_list


# In[5]:


# crossover logic
def crossover(parent1,parent2):
    [startGene,endGene] = sorted(random.choices(range(passcode_length),k=2))
    if random.random()>0.5:
        return parent1[:startGene]+parent2[startGene:endGene]+parent1[endGene:]
    else:
        return parent2[:startGene]+parent1[startGene:endGene]+parent2[endGene:]


# reproduction (gen -1) * population
def create_children(all_parents):
    children = []
    for i in range(len(population)):
        parent1 = all_parents[int(random.random() * len(all_parents))]
        parent2 = all_parents[int(random.random() * len(all_parents))]
        children.append(crossover(parent1,parent2))
    return children


# In[6]:


def mutation(children_set):
    for child in children_set:
        if random.random() < mutation_rate:
            #mutate if rate is less 20 percent 
            child[int(random.random() * passcode_length)] = random.choice(passcode_options)
    return children_set


# In[7]:


# password_option = 1

# population_size = 1000
# num_parents = 20
# mutation_rate = 0.2
# passcode_options = list(string.ascii_uppercase+string.digits+'_')
# passcode_length = 10

# population = [random.choices(passcode_options,k=passcode_length) for i in range(population_size)]
# selection_method = 'tournament'


# fitness_tracker = []
# fitness_scores_list = []
# generations = 0
# t_start = time.time()
# while True:
#     fitness_scores = fitness(population,190573735,password_option)
#     fitness_scores_list.append(fitness_scores)
#     if max([i[1] for i in fitness_scores]) == passcode_length:
#         time_taken = time.time() - t_start
#         password_resolved = ''.join([i[0] for i in fitness_scores if i[1] == passcode_length][0])        
#         print("Passwod resolved in {} generations and {} seconds! \nDiscovered passcode = {}".format(generations,time_taken,password_resolved))
#         break
#     parents = select_parents(fitness_scores,method=selection_method)
#     children = create_children(parents)
#     population = mutation(children)
#     generations += 1    
    
# fitness_tracker = [max([i[1] for i in fitness_scores_list[j]]) for j in range(len(fitness_scores_list))]        
# fig = plt.figure()
# plt.plot(list(range(generations+1)), fitness_tracker)
# fig.suptitle('Fitness Score by Generation', fontsize=14, fontweight='bold')
# ax = fig.add_subplot(111)
# ax.set_xlabel('Generation')
# ax.set_ylabel('Fitness Score')
# plt.savefig('images/'+str(password_option)+'_'+str(population_size)+'_'+str(num_parents)+'_'+str(mutation_rate)+'_'+str(selection_method)+'-'+str(generations)+'-'+str(round(time_taken,3))+'.png',dpi=300, bbox_inches='tight')
# plt.show()  

    
    


# In[8]:


# from sklearn.model_selection import ParameterGrid
# param_grid = {'population_size': [100, 250, 500, 750, 1000, 2000, 3000],
#               'num_parents' : [5, 10, 15, 20, 25, 35, 50],
#               'mutation_rate':[0.1,0.3,0.5,0.8,0.9,1],
#               'selection_method':['rank','mixed_rank','tournament']
#              }

# grid = ParameterGrid(param_grid)
# password_option = 1
# passcode_options = list(string.ascii_uppercase+string.digits+'_')
# passcode_length = 10
# f = open('grid_search.csv','a')
# for i,params in enumerate(grid):
#     population_size = params['population_size']
#     num_parents = params['num_parents']
#     mutation_rate = params['mutation_rate']
#     selection_method = params['selection_method']
# #     print(population_size, num_parents , mutation_rate, selection_method)
#     print(i)
#     population = [random.choices(passcode_options,k=passcode_length) for i in range(population_size)]
#     fitness_tracker = []
#     fitness_scores_list = []
#     generations = 0
#     t_start = time.time()
#     while True:
#         fitness_scores = fitness(population,190573735,password_option)
#         fitness_scores_list.append(fitness_scores)
#         if max([i[1] for i in fitness_scores]) == passcode_length:
#             time_taken = time.time() - t_start
#             password_resolved = ''.join([i[0] for i in fitness_scores if i[1] == passcode_length][0])        
#             f.write(str(password_option)+', '+str(population_size)+', '+str(num_parents)+', '+str(mutation_rate)+', '+str(selection_method)+', '+str(generations)+', '+str(round(time_taken,3))+'\n')
# #             print("Passwod resolved in {} generations and {} seconds! \nDiscovered passcode = {}".format(generations,time_taken,password_resolved))
#             break
#         parents = select_parents(fitness_scores,method=selection_method)
#         children = create_children(parents)
#         population = mutation(children)
#         generations += 1    

# f.close()
    
    


# In[9]:


from sklearn.model_selection import ParameterGrid
param_grid = {'population_size': [100, 250, 500, 750, 1000, 2000, 3000],
              'num_parents' : [5, 10, 15, 20, 25, 35, 50],
              'mutation_rate':[0.1,0.3,0.5,0.8,0.9,1],
              'selection_method':['rank','mixed_rank','tournament']
             }

grid = ParameterGrid(param_grid)
password_option = 1
passcode_options = list(string.ascii_uppercase+string.digits+'_')
passcode_length = 10
f = open('grid_search.csv','a')
# for i,params in enumerate(grid):

# #     print(population_size, num_parents , mutation_rate, selection_method)
#     print(i)

    
    
    
def score(parameter_tuple):
    grid,file = (parameter_tuple)
    f = open('grid_search_'+file+'.csv','a')
#     print(file)
    for i,params in enumerate(grid):
        population_size = params['population_size']
        num_parents = params['num_parents']
        mutation_rate = params['mutation_rate']
        selection_method = params['selection_method']        
#         f.write(str(password_option)+', '+str(population_size)+', '+str(num_parents)+', '+str(mutation_rate)+', '+str(selection_method))
        population = [random.choices(passcode_options,k=passcode_length) for i in range(population_size)]
        fitness_tracker = []
        fitness_scores_list = []
        generations = 0
        t_start = time.time()
        while True:
            fitness_scores = fitness(population,190573735,password_option)
            fitness_scores_list.append(fitness_scores)
            if max([i[1] for i in fitness_scores]) == passcode_length:
                time_taken = time.time() - t_start
                password_resolved = ''.join([i[0] for i in fitness_scores if i[1] == passcode_length][0]) 
    #             return population_size, num_parents , mutation_rate, selection_method, generations, time_taken
                f.write(str(password_option)+', '+str(population_size)+', '+str(num_parents)+', '+str(mutation_rate)+', '+str(selection_method)+', '+str(generations)+', '+str(round(time_taken,3))+'\n')
    #             print("Passwod resolved in {} generations and {} seconds! \nDiscovered passcode = {}".format(generations,time_taken,password_resolved))
                break
            parents = select_parents(fitness_scores,method=selection_method)
            children = create_children(parents)
            population = mutation(children)
            generations += 1    
    f.close()


# In[10]:


from multiprocessing import Pool, freeze_support 

def multicore_function(function_name,alist):
    '''
    this function accept four arguments
    1.function name: name of that function which will called.
    2.wfname: name of file which will be written.
    3.alist: list of data which we breaks in four parts and assigned to different cor.
    4.otherparam: this is review date file name. it will created when updates are found in user reviews.
    it takes list of data lines and make 4 chunckes(parts) and assigned to different cor
    '''
    freeze_support()
    def chunkify(lst,n):
        return [lst[i::n] for i in range(n)]
    #create a process Pool with 4 processes
    pool = Pool(processes=8)

    part1,part2,part3,part4,part5,part6,part7,part8,part9,part10,part11,part12 = (part for part in chunkify(alist, 12))
#     print(part1[0])
    #map doWork to availble Pool processes
    results = pool.map(function_name, (((part1,'part1.csv')),((part2,'part2.csv')),((part3,'part3.csv')),((part4,'part4.csv')),((part5,'part5.csv')),((part6,'part6.csv')),((part7,'part7.csv')),((part8,'part8.csv')),((part9,'part9.csv')),((part10,'part10.csv')),((part11,'part11.csv')),((part12,'part12.csv'))))
    return results




# In[11]:


multicore_function(score,list(grid))


# In[ ]:




