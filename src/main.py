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

from gurobipy import GRB

# Add the src directory to Python path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

#from data_ops.data_loader import DataLoader
#from opt_model.opt_model import OptModel


#data = DataLoader('../data')

#########################
# Question 1a
#########################

"""OPF_model = OPF_OptimizationProblem(OPF_input_data)
    
OPF_model.run()

OPF_model.display_results()"""

#Run optimization model from opt_model.py
"""""
# Create and run the optimization model
print("Creating optimization model...")
problem = OptModel()  # No parameters needed!

print("Running optimization...")
problem.run()

print("Displaying results...")
problem.display_results()

#print dual variables for all constrains

print("Dual Variables for constraints:")
for t in range(problem.T):
    constr = problem.pv_production_constraints[t]
    print(f"Time {t}: PV Production Dual = {constr.Pi}")
"""

from opt_model.opt_model import OptModel  # Only import what you need

print("="*60)
print("QUESTION 1B: DISCOMFORT ANALYSIS")
print("="*60)

# Question 1b with different discomfort weights
print("\n1. Low Discomfort Weight (α = 0.1) - Focus on Cost")
print("-" * 50)
model_1b_low = OptModel(tariff_scenario='TOU_import_tariff_Radius', question='1b', alpha_discomfort=0.1)
model_1b_low.run()
model_1b_low.display_results()

print("\n2. Medium Discomfort Weight (α = 1.0) - Balanced")
print("-" * 50)
model_1b_med = OptModel(tariff_scenario='TOU_import_tariff_Radius', question='1b', alpha_discomfort=1.0)  
model_1b_med.run()
model_1b_med.display_results()

print("\n3. High Discomfort Weight (α = 10.0) - Focus on Comfort")
print("-" * 50)
model_1b_high = OptModel(tariff_scenario='TOU_import_tariff_Radius', question='1b', alpha_discomfort=10.0)
model_1b_high.run()
model_1b_high.display_results()

# Summary comparison
print("\n" + "="*60)
print("SUMMARY COMPARISON")
print("="*60)
print(f"Low Weight (α=0.1):   Cost = {model_1b_low.results.objective_value:.2f} DKK")
print(f"Medium Weight (α=1.0): Cost = {model_1b_med.results.objective_value:.2f} DKK") 
print(f"High Weight (α=10.0):  Cost = {model_1b_high.results.objective_value:.2f} DKK")

print("\nInsight: Higher α → Lower discomfort but higher costs")
print("         Lower α → Higher discomfort but lower costs")

#from data_ops.data_visualizer import DataVisualizer

# Run complete analysis
#visualizer = DataVisualizer()
#visualizer.run_complete_analysis()