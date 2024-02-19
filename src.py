#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tezcan.b

Source code for Surplus Unit Allocation Problem. Please refer to 
README file for assumptions and the math model. 
"""

import pandas as pd 
import numpy as np 
import gurobipy as gp 
import math 
from scipy.stats import burr12
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_table
import warnings
import webbrowser
warnings.simplefilter(action='ignore', category=FutureWarning)

#creates the object for the math model 
class suaModel:
    def __init__(self, data_loc = '/input_sua.xlsx'):
        
        #read the excel sheet into dataframes 
        self.df_prods = pd.read_excel(data_loc, sheet_name = 'Demand', index_col = 0).transpose() 
        self.df_vars = pd.read_excel(data_loc, sheet_name = 'Demand Variance', index_col = 0) 

        #make sure data is read
        assert self.df_prods is not None and self.df_vars is not None, 'Unable to read data, please make sure the file location is correct.' 
        
        #number of products 
        self.n_prods = self.df_prods.shape[0]
        #set of products 
        self.products = set(range(self.n_prods))
        
    def scenarioSampler(self, nScens = 300, rng = 3):
        '''
        Parameters
        ----------
        nScens : int, 
            Number of scenarios to generate. The default is 300.
        rng : int, optional
            Seed for random number generator for reproducibility. The default is 3.

        Returns
        -------
        None. Stores the scenarios with-in the object

        '''
        #Dictionary of distribution objects 
        dists = {idx: burr12(row['c'], row['d'], row['loc'], row['scale']) for idx, row in self.df_vars.iterrows()}
        
        #a simple function to iterate over each product's distribution and sample from them
        def sampleScenarios(nScens, rng = rng):
            scens = np.zeros([nScens, self.n_prods])
            for idx, row in self.df_prods.iterrows():
                scens[:,idx] = dists[row['Variance group']].rvs(size = nScens, random_state = rng) * row['Demand'] 
            
            #make sure there are no negative demands by truncating them to zero. 
            scens = np.clip(scens, a_min=0, a_max=None)
            return np.round(scens)
        self.scens = sampleScenarios(nScens)
        
    def optimize(self, alpha = 0.10, grb_params = {'NumericFocus': 1, 'InfUnbdInfo': 1}):        
        '''
        Parameters
        ----------
        alpha : float 
        Between 0 and 1. Macro-target parameter The default is 0.10.
        grb_params : Dict
            Dictionary of Gurobi parameters. The default is {'NumericFocus': 1, 'InfUnbdInfo': 1}.

        Creates a gurobi model based on given data, runs a linear program and stores the data. 

        Returns
        -------
        None.

        '''

        #make sure there are scenarios generated
        assert self.scens is not None, 'Please generate some scenario using scenarioSampler method'
        
        #make sure macro target parameter makes sense
        assert alpha < 1 and alpha > 0, 'Please make sure that alpha value is between zero and one (even better if between 0.10 and 0.50)'
         
        self.alpha = alpha 
        #Prepare the model input from the dataframes
        margin = dict(zip(self.products, self.df_prods['Margin'].round(4)))
        cost = dict(zip(self.products, self.df_prods['COGS'].round(1)))
        vars_group = dict(zip(self.products, self.df_prods['Variance group']))
        dem = dict(zip(self.products, self.df_prods['Demand'].round(0)))
        caps_vals = dict(zip(self.products, self.df_prods['Capacity']))
        caps_M = sum(dem.values())*alpha #big M
        caps = dict(zip(self.products, list(caps_M if math.isnan(caps_vals[p]) else 
                    round(dem[p]*caps_vals[p]) for p in self.products)))

        # Use groupby to group by each group and apply a lambda function to get indices
        grouped_indices = self.df_prods.groupby('Substitutability group').apply(lambda x: x.index.tolist())
        # Convert the result to a dictionary
        groups = grouped_indices.to_dict()
        self.groups = groups
        
        #dictionary of outgoing arcs for each node
        O = {i: [-1] + groups[self.df_prods['Substitutability group'][i]] for i in self.products}
        
        #dictionary of incoming arcs for each node
        I = {i: groups[self.df_prods['Substitutability group'][i]] for i in self.products}
            
        #number of scenarios 
        n_scens = self.scens.shape[0]
        S = range(n_scens) #set of scenario indices 
        scen_prob = 1/n_scens #a scenario's probability
        
        #arcs of our network 
        arcs = [(key, value) for key, values in O.items() for value in values]
        
        #gurobi model object 
        m = gp.Model()
        #variables 
        x = m.addVars(self.products, lb = 0)
        y = m.addVars(arcs, n_scens, lb = 0)
        s_aux = m.addVars(groups, lb = 0)
        s_max = m.addVar()
        
        #capacity constraint 
        m.addConstrs(x[i] <= caps[i] for i in self.products)

        #flow balance constraint 
        m.addConstrs(x[i] + dem[i] == sum(y[i,j,w] for j in O[i]) for i in self.products for w in S)
        
        #assignment of products under varying scenarios  
        for w in S:
            for j in self.products:
                m.addConstr(sum(y[i,j,w] for i in I[j]) <= self.scens[w][j])
                #make sure demand is first fulfilled by self 
                m.addConstr(y[j,j,w] >= min(dem[j], self.scens[w][j]))
        
        #stay within surplus production limit 
        m.addConstr(sum(x[p] for p in self.products) <= sum(dem.values())*alpha)

        #calculate the surplus for each group
        for g in groups:
            m.addConstr(s_aux[g] == (sum(y[i,j,w]*scen_prob for i in groups[g] for j in set(O[i]) - set([i] + [-1]) for w in S)))    
        
        #create the min-max variable by upper bounding all group surplus variables
        for i in groups:
            m.addConstr(s_aux[i] <= s_max)
                   
        ##obj 1:profit = revenue - cost: margin*sold - cost*unsold
        #profit
        obj = gp.LinExpr()
        for w in S:
            for i in self.products: 
                for j in set(O[i]) - set([-1]):
                    obj.add(margin[i]*y[i,j,w], scen_prob)
        #cost: cost*unsold 
        for w in S:
            for i in self.products: 
                obj.add(-cost[i]*y[i,-1,w], scen_prob)


        #obj1: profit - higher the priority higher the hierarchy
        m.setObjectiveN(-obj, index = 0, priority = 2, name = 'Max Profit')

        #obj2: minimize the maximum substituion amount 
        m.setObjectiveN(s_max, index = 1, priority = 1, name = 'Min Max Substitute')
        
        #Set gurobi parameters 
        for par in grb_params:
            m.setParam(par, grb_params[par])

        m.optimize()
        
        #list of objective functions
        z = list()
        for o in range(2):
            # Set which objective we will query
            m.params.ObjNumber = o
            # Query the o-th objective value
            z.append(m.ObjNVal)
        
        #store the model and optimal solutions 
        self.model = m
        self.x = x 
        self.y = y
        self.min_max_subs = s_max.x
        self.max_profit = -z[0]

    def printOutputs(self):
        '''
        Post-process optimization results for easier interpretation.
        self.prod_results gives a prettier dataframe with allocation decisions and other stats
        self.obj_results gives a dataframe with objective function results 

        Returns
        -------
        None.

        '''
        #print Outputs to a csv after postprocessing
        x_dict = {v : self.x[v].x for v in self.x}
        y_dict = {v: self.y[v].x for v in self.y}

        obj_results = pd.DataFrame()
        obj_results['Alpha'] = [self.alpha]
        obj_results['Expected Profit'] = [self.max_profit]
        obj_results['Expected Minimum Maximum Substitition'] = [self.min_max_subs]
        self.obj_results = obj_results
        obj_results.to_csv(f"results_{self.alpha*10}.csv")

        #product df 
        prod_results_df = pd.DataFrame.from_dict(x_dict, orient = 'index', columns = ['Surplus'])
        prod_results_df['Product'] = list(range(0, self.n_prods))
        prod_results_df['Base Production'] = self.df_prods['Demand']
        
        y_array = np.zeros((self.n_prods, self.n_prods + 1, self.scens.shape[0]), dtype=object)
        # Fill the array with values from the dictionary
        for (i, j, w), value in y_dict.items():
            y_array[i, j, w] = value
                        
        prod_results_df['Average Unused Stock'] = np.round(np.mean(y_array[:,-1,:], axis = 1).astype('float'), 2)
        prod_results_df['Alpha'] = np.ones([self.n_prods])*self.alpha
        ave_assign_pair = np.sum(y_array[:,:-1,:], axis = 2)/self.scens.shape[0]
        ave_assign_to_others = (np.sum(ave_assign_pair, axis = 1) - np.diag(ave_assign_pair))
        prod_results_df['Ave_Assign_to_Others'] = pd.DataFrame(np.round(ave_assign_to_others.astype('float'), 2))   
        prod_results_df['Ave_Self_Assign'] = pd.DataFrame(np.round(np.diag(ave_assign_pair).astype('float'), 2))
        prod_results_df['Substitutability Group'] = self.df_prods['Substitutability group']
        prod_results_df = prod_results_df.loc[:,['Product', 'Surplus', 'Base Production',
                              'Substitutability Group', 'Ave_Self_Assign', 'Ave_Assign_to_Others',
                              'Average Unused Stock', 'Alpha']]
        prod_results_df.to_csv(f'sua_{self.alpha*100}.csv')
        self.prod_results = prod_results_df
        #group df 
        ave_group_sub = {}
        for g in self.groups:
            ave_group_sub[g] = sum(ave_assign_to_others[self.groups[g]])
        groups_df = pd.DataFrame.from_dict(ave_group_sub, orient = 'index', columns = ['Ave_Substition'])
        groups_df['Group Surplus'] = prod_results_df.groupby('Substitutability Group').sum()['Surplus']
        self.groups_df = groups_df 
        
def dash_table_maker(experiment_results, opt_solns):
    '''
    Gets optimal solution results and objective functions and creates an interactive dashtable. 
    Parameters
    ----------
    experiment_results : pd.DataFrame 
    opt_solns : pd.DataFrame

    '''
    # Initialize Dash app
    app = dash.Dash(__name__)
    
    # Define app layout
    app.layout = html.Div([
        # Dropdown menu for selecting alpha value
        dcc.Dropdown(
            id='alpha-dropdown',
            options=[{'label': str(alpha), 'value': alpha} for alpha in experiment_results['Alpha'].unique()],
            value=experiment_results['Alpha'].iloc[0],
            style={'width': '50%'}
        ),
    
        # Display final value based on selected alpha
        html.Div(id='final-value'),
    
        # DataTable to display experiment results
        dash_table.DataTable(
            id='table',
            columns=[
                {'name': col, 'id': col, 'deletable': False} for col in experiment_results.columns
            ],
            style_table={'overflowX': 'auto'},
            editable=True,
            filter_action='native',
            sort_action='native',
            sort_mode='multi',
            row_selectable='multi',
            row_deletable=True,
            selected_rows=[],
        ),
    ])
    
    # Define callback to update final value and table based on selected alpha
    @app.callback(
        [Output('final-value', 'children'),
         Output('table', 'data')],
        [Input('alpha-dropdown', 'value')]
    )
    def update_content(selected_alpha):
        # Calculate and display the final value
        final_value = f"Profit: {opt_solns.loc[opt_solns['Alpha'] == selected_alpha, 'Expected Profit'].values[0]}"
    
        # Filter the data for the DataTable based on the selected alpha
        filtered_data = experiment_results[experiment_results['Alpha'] == selected_alpha]
        table_data = filtered_data.to_dict('records')
    
        return final_value, table_data
    
    # Run the app
    if __name__ == '__main__':
        app.run_server(debug=True, use_reloader=False, port=8050)

    webbrowser.open('http://127.0.0.1:8050/')
