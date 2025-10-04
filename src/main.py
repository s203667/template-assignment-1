"""
Placeholder for main function to execute the model runner. This function creates a single/multiple instance of the Runner class, prepares input data,
and runs a single/multiple simulation.

Suggested structure:
- Import necessary modules and functions.
- Define a main function to encapsulate the workflow (e.g. Create an instance of your the Runner class, Run a single simulation or multiple simulations, Save results and generate plots if necessary.)
- Prepare input data for a single simulation or multiple simulations.
- Execute main function when the script is run directly.
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from data_ops.data_loader import DataLoader


data = DataLoader('../data')

#########################
# Question 1a
#########################

app_data1a = data._load_data_file('question_1a', 'appliance_params.json')
# First, let's see what's actually in the dictionary

print(app_data1a['DER'][0]['max_power_kW'])

"""
# This corresponds to the main function
input_data = InputData(
    VARIABLES = ['x1', 'x2'],
    objective_coeff = {'x1': 30, 'x2': 20},
    constraints_coeff = {'x1': [0.6, 0.4], 'x2': [0.2, 0.8]},
    constraints_rhs = [60, 100],
    constraints_sense =  [GRB.GREATER_EQUAL, GRB.GREATER_EQUAL],
)
problem = LP_OptimizationProblem(input_data)
problem.run()
problem.display_results()
"""