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
# =============================================================================
# RUN OPTIMIZATION MODEL
# =============================================================================

# Create and run model - change parameters as needed
model = OptModel(tariff_scenario='import_tariff', question='1c', alpha_discomfort=2, consumer_type='original')
model.run()
model.display_results()

# Create visualizer for plotting
visualizer = DataVisualizer(model)

# =============================================================================
# PLOTTING OPTIONS (UNCOMMENT AS NEEDED)
# =============================================================================

"""
# Question 1c plots
fig1, data1 = visualizer.plot_dual_sensitivity_analysis_1c()      # 1a-style + battery
fig2, data2 = visualizer.plot_alpha_sensitivity_analysis_1c()     # 1b-style + battery
visualizer.show_plot()
"""

"""
# Battery profitability analysis (Question 2b)
profitability_results = visualizer.conduct_battery_profitability_experiment(battery_cost_per_kwh=3000)
"""

"""
# Question 1a plots (change model to question='1a' above)
# fig_duals = visualizer.plot_dual_sensitivity_analysis()
# visualizer.show_plot()
"""

"""
# Question 1b plots (change model to question='1b' above)
# fig_alpha, alpha_data = visualizer.plot_alpha_sensitivity_analysis_1b()
# visualizer.show_plot()
"""