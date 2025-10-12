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
from data_ops.data_visualizer import DataVisualizer
model_1a = OptModel(tariff_scenario='import_tariff', question='1a', alpha_discomfort=2, consumer_type='original')
model_1a.run()
model_1a.display_results()

# Create dual variables analysis plot
print("\nCreating dual variables analysis plot...")
visualizer = DataVisualizer(model_1a)

# Option 1: Basic dual variables plot
fig_duals = visualizer.plot_dual_sensitivity_analysis()
#visualizer.save_plot(fig_duals, 'dual_variables_analysis.png')


# Show plots
visualizer.show_plot()


"""
# Create visualizer and plot results
from data_ops.data_visualizer import DataVisualizer
print("\nCreating visualization...")
visualizer = DataVisualizer(model_1a)
fig = visualizer.plot_question_1a_results()
"""
# Save and show plot
#visualizer.save_plot(fig, 'question_1a_results.png')
#visualizer.show_plot()

# Create stacked comparison plots
#print("\nCreating stacked comparison plots...")


# Option 2: Optimized stacked plot (runs optimization for each scenario)

#fig_optimized = visualizer.plot_optimized_stacked_comparison() 

#visualizer.save_plot(fig_optimized, 'optimized_stacked_comparison.png')

# Show both plots
#visualizer.show_plot()

