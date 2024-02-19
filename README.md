
Please see the Project and Model Description document for a detailed description of my approach. This document is about how to use the source code for the math model and output generation. 

My code consists of a source code file, src.py, and a main file, main.py, which can be accessed from this GitHub repository. Required Python packages can be imported by using requirements.txt or by manually importing them. They can be found on top of the src.py. 

If you want to run the model and get results without modifying any settings (which you can) you should check main.py.  You would only need to specify where the input Excel sheet is, and the code will create a model object and solve it for the listed values of alpha_list. You can change that list as well based on the needs. This code piece will 1) create a pd.DataFrame with results, 2) create a CSV for each alpha value with results, and 3) will create a dashboard on your web browser where you can see the results interactively by using a dash table. 

If you would like to see how the model works, please check the src.py. The code is commented and modular. There 3 methods besides initialization. 

•	scenarioSampler(): generates scenarios based on the given distributions. Number of scenarios and rng can be changed. 
•	optimize(): creates a Gurobi model object based on the input and the model defined in the other document. Alpha value and Gurobi parameters can be changed. 
•	printOutputs(): creates a dataframe from the optimal solution and also writes solutions to a csv file. 

There is also an additional dash_table_maker function, which takes experiment results dataframe and objective function results dataframe and creates a browser-based dashboard. 

