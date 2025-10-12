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



from opt_model.opt_model import OptModel  # Only import what you need
from data_ops.data_visualizer import DataVisualizer
#model_1b = OptModel(tariff_scenario='import_tariff', question='1b', alpha_discomfort=2, consumer_type='original')
#model_1b.run()
#model_1b.display_results()
"""
model_1c = OptModel(tariff_scenario='import_tariff', question='1c', alpha_discomfort=2, consumer_type='original')
model_1c.run()

visualizer_1c = DataVisualizer(model_1c)

# Use the new Question 1c methods
fig1, data1 = visualizer_1c.plot_dual_sensitivity_analysis_1c()      # 1a-style + battery
fig2, data2 = visualizer_1c.plot_alpha_sensitivity_analysis_1c()     # 1b-style + battery

visualizer_1c.show_plot()
"""
""""""
model_base = OptModel(tariff_scenario='import_tariff', question='1c', alpha_discomfort=1.5, consumer_type='original')
visualizer = DataVisualizer(model_base)

# Run comprehensive profitability analysis
profitability_results = visualizer.conduct_battery_profitability_experiment(battery_cost_per_kwh=3000)



"""
################################
#PLOTS FOR 1.a
################################
# Create dual variables analysis plot
print("\nCreating dual variables analysis plot...")
visualizer = DataVisualizer(model_1a)

# Option 1: Basic dual variables plot
fig_duals = visualizer.plot_dual_sensitivity_analysis()
#visualizer.save_plot(fig_duals, 'dual_variables_analysis.png')


# Show plots
visualizer.show_plot()

"""

"""
################################
#plots for 1b
################################

print("\nCreating alpha sensitivity analysis for Question 1b...")
visualizer = DataVisualizer(model_1b)

fig_alpha, alpha_data = visualizer.plot_alpha_sensitivity_analysis_1b()
visualizer.show_plot()
"""
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

