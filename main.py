#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tezcan.b

Illustrates how to use the src.py to solve SUA problem. 
"""

from src import suaModel, dash_table_maker 
import pandas as pd 


#MAKE SURE TO UPDATE THE INPUT LOCATION  
data_loc = '/SUA/input_sua.xlsx'
#creates the model object
m = suaModel(data_loc)
#samples scenarios 
m.scenarioSampler()

#change the alpha_list for the parameters you want to explore 
alpha_list = [0.10, 0.20, 0.30, 0.40, 0.50]
#initialize dataframes for bookkeeping
results_df = pd.DataFrame()
opt_sols = pd.DataFrame()
#solve for each alpha in alpha_list
for a in alpha_list:
    m.optimize(alpha = a)
    m.printOutputs()
    df_temp = m.prod_results
    results_temp = m.obj_results
    results_df = pd.concat([results_df, df_temp])
    opt_sols = pd.concat([opt_sols, results_temp])

#make a dash table 
dash_table_maker(results_df, opt_sols)
