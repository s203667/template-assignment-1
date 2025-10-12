import json
import csv
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys
from opt_model.opt_model import OptModel

# Add src to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

class DataVisualizer:
    
    def __init__(self, model):
        """
        Initialize DataVisualizer with an OptModel instance
        
        Args:
            model: OptModel instance containing the optimization results and data
        """
        self.model = model

    def plot_dual_variables_analysis(self, figsize=(14, 10)):
        """
        Plot dual variables for energy balance constraints together with:
        - Hourly demand (D_hour)
        - Max D_hour limit line
        - Hourly import
        - Hourly PV production
        
        Args:
            figsize: Figure size tuple (width, height)
        """
        # Create figure with dual y-axes
        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = ax1.twinx()  # Second y-axis for dual variables
        
        # Hours for x-axis
        hours = list(range(24))
        
        # Extract data from model results
        if hasattr(self.model, 'results'):
            hourly_demand = [self.model.results.D_hour[t] for t in range(24)]
            hourly_import = [self.model.results.P_imp[t] for t in range(24)]
            hourly_pv = [self.model.results.P_PV_prod[t] for t in range(24)]
        else:
            print("Warning: No results found in model")
            hourly_demand = [0] * 24
            hourly_import = [0] * 24
            hourly_pv = [0] * 24
        
        # Extract dual variables for energy balance constraints
        dual_values = []
        if hasattr(self.model, 'energy_balance_constraints'):
            for t in range(24):
                # Handle different constraint naming schemes
                if t in self.model.energy_balance_constraints:
                    dual_values.append(self.model.energy_balance_constraints[t].Pi)
                elif f'upper_{t}' in self.model.energy_balance_constraints:
                    dual_values.append(self.model.energy_balance_constraints[f'upper_{t}'].Pi)
                else:
                    dual_values.append(0)
        else:
            dual_values = [0] * 24

            # Get price data
        energy_prices = getattr(self.model, 'energy_prices', [0] * 24)
        active_tariff = getattr(self.model, 'active_tariff', [0] * 24)
        
        # Calculate combined prices (energy price + tariff)
        combined_prices = [ep + tariff for ep, tariff in zip(energy_prices, active_tariff)]
    
        # Get max D_hour limit
        max_d_hour = getattr(self.model, 'max_power_load', 5.0)  # Default if not found
        
        # Plot power data on left y-axis (kW)
        line1 = ax1.step(hours, hourly_demand, 'purple', linewidth=3, where='mid',
                        label='Hourly Demand (D_hour)', linestyle='-')
        line2 = ax1.step(hours, hourly_import, 'blue', linewidth=2, where='mid',
                        label='Hourly Import', linestyle='-')
        line3 = ax1.step(hours, hourly_pv, 'orange', linewidth=2, where='mid',
                        label='PV Production', linestyle='-')
        
        # Add max D_hour limit as horizontal line
        line4 = ax1.axhline(y=max_d_hour, color='red', linewidth=2, linestyle='--',
                        label=f'Max D_hour Limit ({max_d_hour:.1f} kW)')
        
        # Plot dual variables on right y-axis (DKK/kWh)
        line5 = ax2.step(hours, dual_values, 'black', linewidth=3, where='mid',
                        label='Energy Balance Duals', linestyle=':', marker='o', markersize=4)
        line6 = ax2.step(hours, combined_prices, 'magenta', linewidth=2, where='mid',
                        label='Total Price (Energy + Tariff)', linestyle='--', marker='s', markersize=3)
    
        # Add zero line for dual variables
        ax2.axhline(y=0, color='gray', linewidth=1, linestyle=':', alpha=0.7)
        
        # Formatting
        ax1.set_xlabel('Hours', fontsize=12)
        ax1.set_ylabel('Power (kW)', fontsize=12, color='black')
        ax2.set_ylabel('Dual Variables & Prices + Tariffs (DKK/kWh)', fontsize=12, color='green')
        
        # Set x-axis
        ax1.set_xlim(-0.5, 23.5)
        ax1.set_xticks(range(0, 24, 2))
        ax1.set_xticklabels([f'{h}:00' for h in range(0, 24, 2)], rotation=45)
        
        # Grid
        ax1.grid(True, alpha=0.3)
        
        # Color y-axis labels
        ax1.tick_params(axis='y', labelcolor='black')
        ax2.tick_params(axis='y', labelcolor='black')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
        
        # Title
        title = 'Energy Balance Analysis: Dual Variables and Power Flows'
        plt.title(title, fontsize=14, fontweight='bold')
        
        # Tight layout
        plt.tight_layout()
        
        # Print summary of dual variables
        print("\nDual Variables Summary:")
        print(f"  Maximum dual: {max(dual_values):.4f} DKK/kWh (Hour {dual_values.index(max(dual_values))})")
        print(f"  Minimum dual: {min(dual_values):.4f} DKK/kWh (Hour {dual_values.index(min(dual_values))})")
        print(f"  Average dual: {sum(dual_values)/len(dual_values):.4f} DKK/kWh")
        
        # Find hours with significant dual values
        significant_duals = [(i, val) for i, val in enumerate(dual_values) if abs(val) > 0.01]
        if significant_duals:
            print(f"  Hours with significant duals (>0.01):")
            for hour, dual in significant_duals:
                print(f"    Hour {hour}: {dual:.4f} DKK/kWh")
        
        return fig
    def plot_dual_sensitivity_analysis(self, figsize=(18, 12)):
        """
        Create three plots showing how dual variables change with different tariff/price scenarios:
        1. Fixed tariff + Fixed energy price
        2. Fixed tariff + Dynamic energy price  
        3. Dynamic tariff + Fixed energy price
        
        Args:
            figsize: Figure size tuple (width, height)
        """
        from opt_model.opt_model import OptModel
        
        # Get data from bus_params.json
        fixed_import_tariff = 0.5  # import_tariff_DKK/kWh from your data
        fixed_energy_price = 1.0   # fixed_energy_price_DKK_per_kWh from your data
        
        # Dynamic energy prices from your data
        dynamic_energy_prices = [
            1.1, 1.05, 1.0, 0.9, 0.85, 1.01, 1.05, 1.2, 1.4, 1.6,
            1.5, 1.1, 1.05, 1.0, 0.95, 1.0, 1.2, 1.5, 2.1, 2.5,
            2.2, 1.8, 1.4, 1.2
        ]
        
        # Dynamic import tariff from your data (TOU Radius)
        dynamic_import_tariff = [
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.29, 0.29, 0.29, 0.29,
            0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.88, 0.88, 0.88,
            0.88, 0.29, 0.29, 0.29
        ]
        
        # Define three scenarios based on your data
        price_scenarios = [
            {
                'name': 'Scenario 1: Fixed Tariff + Fixed Energy Price',
                'description': f'Fixed import tariff ({fixed_import_tariff} DKK/kWh) + Fixed energy price ({fixed_energy_price} DKK/kWh)',
                'energy_prices': [fixed_energy_price] * 24,
                'import_tariff': [fixed_import_tariff] * 24,
                'tariff_scenario': 'import_tariff'  # Use flat tariff scenario
            },
            {
                'name': 'Scenario 2: Fixed Tariff + Dynamic Energy Price',
                'description': f'Fixed import tariff ({fixed_import_tariff} DKK/kWh) + Dynamic energy prices',
                'energy_prices': dynamic_energy_prices,
                'import_tariff': [fixed_import_tariff] * 24,
                'tariff_scenario': 'import_tariff'  # Use flat tariff scenario
            },
            {
                'name': 'Scenario 3: Dynamic Tariff + Fixed Energy Price',
                'description': f'TOU Radius tariff + Fixed energy price ({fixed_energy_price} DKK/kWh)',
                'energy_prices': [fixed_energy_price] * 24,
                'import_tariff': dynamic_import_tariff,
                'tariff_scenario': 'TOU_import_tariff_Radius'  # Use TOU tariff scenario
            }
        ]
        
        # Create subplots - 3 scenarios vertically
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        fig.suptitle('Dual Variable Sensitivity Analysis: Fixed vs Dynamic Tariffs and Energy Prices', 
                    fontsize=16, fontweight='bold')
        
        hours = list(range(24))
        scenario_results = {}
        
        for idx, scenario in enumerate(price_scenarios):
            print(f"\nRunning {scenario['name']}")
            
            # Create new model with specified scenario
            temp_model = OptModel(
                tariff_scenario=scenario['tariff_scenario'],
                question=self.model.question,
                alpha_discomfort=getattr(self.model, 'alpha_discomfort', 1.0),
                consumer_type='original'
            )
            
            # Override energy prices and tariff
            temp_model.energy_prices = scenario['energy_prices']
            temp_model.active_tariff = scenario['import_tariff']
            
            # Build and run optimization
            temp_model._build_variables()
            temp_model._build_constraints()
            temp_model._build_objective_function()
            temp_model.model.update()
            temp_model.model.optimize()
            
            if temp_model.model.status == 2:  # Optimal
                temp_model._save_results()
                
                # Extract data
                hourly_demand = [temp_model.results.D_hour[t] for t in range(24)]
                hourly_import = [temp_model.results.P_imp[t] for t in range(24)]
                hourly_pv = [temp_model.results.P_PV_prod[t] for t in range(24)]
                
                # Extract dual variables
                dual_values = []
                for t in range(24):
                    if t in temp_model.energy_balance_constraints:
                        dual_values.append(temp_model.energy_balance_constraints[t].Pi)
                    elif f'upper_{t}' in temp_model.energy_balance_constraints:
                        dual_values.append(temp_model.energy_balance_constraints[f'upper_{t}'].Pi)
                    else:
                        dual_values.append(0)
                
                # Calculate combined prices
                combined_prices = [ep + tariff for ep, tariff in zip(scenario['energy_prices'], scenario['import_tariff'])]
                
                # Store results
                scenario_results[scenario['name']] = {
                    'dual_values': dual_values,
                    'combined_prices': combined_prices,
                    'energy_prices': scenario['energy_prices'],
                    'import_tariff': scenario['import_tariff'],
                    'hourly_demand': hourly_demand,
                    'hourly_import': hourly_import,
                    'hourly_pv': hourly_pv,
                    'objective_value': temp_model.results.objective_value,
                    'daily_demand_dual': temp_model.daily_demand_constraint.Pi if hasattr(temp_model, 'daily_demand_constraint') else 0.0

                }
                
                # Create subplot
                ax1 = axes[idx]
                ax2 = ax1.twinx()
                
                # Get max D_hour limit
                max_d_hour = getattr(temp_model, 'max_power_load', 5.0)
                
                # Plot power data on left y-axis (kW)
                ax1.step(hours, hourly_demand, 'blue', linewidth=3, where='mid',
                        label='Hourly Demand (D_hour)', linestyle=':')
                ax1.step(hours, hourly_import, 'red', linewidth=2, where='mid',
                        label='Hourly Import', linestyle='-')
                ax1.step(hours, hourly_pv, 'yellow', linewidth=2, where='mid',
                        label='PV Production', linestyle='-')
                
                # Add max D_hour limit as horizontal line
                max_d_hour = temp_model.max_power_load
                ax1.axhline(y=max_d_hour, color='turquoise', linewidth=2, linestyle='--',
                        label=f'Max D_hour Limit ({max_d_hour:.1f} kW)')
                
                # Plot dual variables and prices on right y-axis
                ax2.step(hours, dual_values, 'black', linewidth=2, where='mid',
                        label='Energy Balance Duals', linestyle='-')
                ax2.step(hours, combined_prices, 'magenta', linewidth=2, where='mid',
                        label='Total Price (Energy + Tariff)', linestyle='--', marker='s', markersize=3)
                
                # Add zero line for dual variables
                ax2.axhline(y=0, color='gray', linewidth=1, linestyle=':', alpha=0.7)
                
                # Formatting
                ax1.set_ylabel('Power (kW)', fontsize=11)
                ax2.set_ylabel('Dual Variables & Prices (DKK/kWh)', fontsize=11, color='black')
                
                # Title with scenario description and cost
                ax1.set_title(f'{scenario["name"]}\nTotal Cost: {temp_model.results.objective_value:.2f} DKK', 
                            fontsize=11, fontweight='bold')
                
                ax1.set_xlim(-0.5, 23.5)
                ax1.grid(True, alpha=0.3)
                
                # Color y-axis labels
                ax1.tick_params(axis='y', labelcolor='black')
                ax2.tick_params(axis='y', labelcolor='black')
                
                # Legend
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
                
            else:
                print(f" Optimization failed for {scenario['name']}")
                axes[idx].text(0.5, 0.5, f'Optimization Failed\nfor {scenario["name"]}', 
                            transform=axes[idx].transAxes, ha='center', va='center', fontsize=14)
        
        # X-axis formatting
        axes[-1].set_xlabel('Hours', fontsize=12)
        axes[-1].set_xticks(range(0, 24, 2))
        axes[-1].set_xticklabels([f'{h}:00' for h in range(0, 24, 2)], rotation=45)
        
        plt.tight_layout()
        
        # Print detailed comparison summary
        print(f"\n" + "="*90)
        print("TARIFF & ENERGY PRICE SCENARIO COMPARISON")
        print("="*90)
        
        for scenario_name, results in scenario_results.items():
            print(f"\n{scenario_name}:")
            print(f"  Total Cost: {results['objective_value']:.2f} DKK")
            print(f"  Total Import: {sum(results['hourly_import']):.2f} kW")
            print(f"  Peak Import Hour: {results['hourly_import'].index(max(results['hourly_import']))} (Import: {max(results['hourly_import']):.2f} kW)")
            print(f"  Peak Demand Hour: {results['hourly_demand'].index(max(results['hourly_demand']))} (Demand: {max(results['hourly_demand']):.2f} kW)")
            print(f"  Dual Variables:")
            print(f"    Max: {max(results['dual_values']):.4f} DKK/kWh (Hour {results['dual_values'].index(max(results['dual_values']))})")
            print(f"    Min: {min(results['dual_values']):.4f} DKK/kWh (Hour {results['dual_values'].index(min(results['dual_values']))})")
            print(f"    Avg: {sum(results['dual_values'])/len(results['dual_values']):.4f} DKK/kWh")
            print(f"  Price Structure:")
            print(f"    Max Total Price: {max(results['combined_prices']):.4f} DKK/kWh (Hour {results['combined_prices'].index(max(results['combined_prices']))})")
            print(f"    Min Total Price: {min(results['combined_prices']):.4f} DKK/kWh (Hour {results['combined_prices'].index(min(results['combined_prices']))})")
            print(f"  Daily Demand Constraint Dual: {results['daily_demand_dual']:.4f}")
        return fig, scenario_results



    def plot_alpha_sensitivity_analysis_1b(self, figsize=(18, 12)):
        """
        Analyze how alpha_discomfort affects the trade-off between cost and comfort in Question 1b.
        Uses data from question_1b bus_params.json
        
        Scenario 1: Alpha = avg(energy_price + tariff) - Equal weight to cost and comfort
        Scenario 2: Alpha = 0.1 * alpha_base - Price matters more (low comfort penalty)
        Scenario 3: Alpha = 10 * alpha_base - Comfort matters more (high comfort penalty)
        """
        from opt_model.opt_model import OptModel
        from data_ops.data_loader import DataLoader
        
        # Load Question 1b data
        data_loader_1b = DataLoader('../data')
        data_loader_1b._load_data('question_1b', 'original')
        
        # Get energy prices and tariff from question_1b data
        energy_prices_1b = data_loader_1b.energy_prices
        import_tariff_1b = data_loader_1b.import_tariff
        
        # Calculate base alpha (equal weighting with 1b prices)
        avg_total_price = sum(energy_prices_1b[t] + import_tariff_1b for t in range(24)) / 24
        
        alpha_scenarios = [
            {
                'name': 'Scenario 1: Equal Weight (Cost ≈ Comfort)',
                'description': f'Alpha = {avg_total_price:.3f} (avg total price from 1b) - Equal cost/comfort sensitivity',
                'alpha_value': avg_total_price
            },
            {
                'name': 'Scenario 2: Price Priority', 
                'description': f'Alpha = {avg_total_price * 0.85:.3f} - Low comfort penalty, price matters more',
                'alpha_value': avg_total_price * 0.85
            },
            {
                'name': 'Scenario 3: Comfort Priority',
                'description': f'Alpha = {avg_total_price * 1.3:.3f} - High comfort penalty, stay close to reference',
                'alpha_value': avg_total_price * 1.3
            }
        ]
        
        # Create subplots - 3 scenarios vertically
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        fig.suptitle('Alpha Sensitivity Analysis (Question 1b): Cost vs Comfort Trade-off', 
                    fontsize=16, fontweight='bold')
        
        hours = list(range(24))
        scenario_results = {}
        
        for idx, scenario in enumerate(alpha_scenarios):
            print(f"\nRunning {scenario['name']} with alpha = {scenario['alpha_value']:.3f}")
            
            # Create new model with specified alpha
            temp_model = OptModel(
                tariff_scenario='import_tariff',  # Use flat tariff from 1b
                question='1b',
                alpha_discomfort=scenario['alpha_value'],
                consumer_type='original'
            )
            
            # Override with question 1b data to ensure consistency
            temp_model.energy_prices = energy_prices_1b
            temp_model.active_tariff = [import_tariff_1b] * 24  # Flat tariff
            
            # Build and run optimization
            temp_model._build_variables()
            temp_model._build_constraints()
            temp_model._build_objective_function()
            temp_model.model.update()
            temp_model.model.optimize()
            
            if temp_model.model.status == 2:  # Optimal
                temp_model._save_results()
                
                # Extract data
                hourly_demand = [temp_model.results.D_hour[t] for t in range(24)]
                hourly_import = [temp_model.results.P_imp[t] for t in range(24)]
                hourly_pv = [temp_model.results.P_PV_prod[t] for t in range(24)]
                
                # Calculate reference demand for comparison
                hourly_reference = [temp_model.hourly_preference[t] * temp_model.max_power_load for t in range(24)]
                
                # Extract dual variables
                dual_values = []
                for t in range(24):
                    if t in temp_model.energy_balance_constraints:
                        dual_values.append(temp_model.energy_balance_constraints[t].Pi)
                    else:
                        dual_values.append(0)
                
                # Calculate combined prices
                combined_prices = [energy_prices_1b[t] + import_tariff_1b for t in range(24)]
                
                # Calculate discomfort metrics
                total_discomfort = temp_model.results.total_discomfort
                import_cost = sum(temp_model.results.P_imp[t] * combined_prices[t] for t in range(24))
                discomfort_cost = scenario['alpha_value'] * total_discomfort
                
                # Store results
                scenario_results[scenario['name']] = {
                    'alpha_value': scenario['alpha_value'],
                    'dual_values': dual_values,
                    'combined_prices': combined_prices,
                    'hourly_demand': hourly_demand,
                    'hourly_reference': hourly_reference,
                    'hourly_import': hourly_import,
                    'hourly_pv': hourly_pv,
                    'objective_value': temp_model.results.objective_value,
                    'total_discomfort': total_discomfort,
                    'import_cost': import_cost,
                    'discomfort_cost': discomfort_cost
                }
                
                # Create subplot
                ax1 = axes[idx]
                ax2 = ax1.twinx()
                
                # Get max D_hour limit
                max_d_hour = temp_model.max_power_load
                
                # Plot power data on left y-axis (kW)
                ax1.step(hours, hourly_demand, 'blue', linewidth=3, where='mid',
                        label='Actual Demand (D_hour)', linestyle='-')
                ax1.step(hours, hourly_reference, 'steelblue', linewidth=2, where='mid',
                        label='Reference Demand', linestyle=':', alpha=0.7)
                ax1.step(hours, hourly_import, 'red', linewidth=2, where='mid',
                        label='Hourly Import', linestyle='-')
                ax1.step(hours, hourly_pv, 'yellow', linewidth=2, where='mid',
                        label='PV Production', linestyle='-')
                
                # Add max D_hour limit as horizontal line
                ax1.axhline(y=max_d_hour, color='turquoise', linewidth=2, linestyle='--',
                        label=f'Max D_hour Limit ({max_d_hour:.1f} kW)')
                
                # Plot dual variables and prices on right y-axis
                ax2.step(hours, dual_values, 'black', linewidth=2, where='mid',
                        label='Energy Balance Duals', linestyle='-')
                ax2.step(hours, combined_prices, 'magenta', linewidth=2, where='mid',
                        label='Total Price (Energy + Tariff)', linestyle='--', marker='s', markersize=3)
                
                # Add zero line for dual variables
                ax2.axhline(y=0, color='gray', linewidth=1, linestyle=':', alpha=0.7)
                
                # Formatting
                ax1.set_ylabel('Power (kW)', fontsize=11)
                ax2.set_ylabel('Dual Variables & Prices (DKK/kWh)', fontsize=11, color='black')
                
                # Title with scenario description and costs breakdown
                ax1.set_title(f'{scenario["name"]}\n'
                            f'Optimal Objective: {temp_model.results.objective_value:.2f}  '
                            f'(Import: {import_cost:.2f} + Discomfort: {discomfort_cost:.2f})', 
                            fontsize=11, fontweight='bold')
                
                ax1.set_xlim(-0.5, 23.5)
                ax1.grid(True, alpha=0.3)
                
                # Color y-axis labels
                ax1.tick_params(axis='y', labelcolor='black')
                ax2.tick_params(axis='y', labelcolor='black')
                
                # Legend
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
                
            else:
                print(f"Optimization failed for {scenario['name']}")
                axes[idx].text(0.5, 0.5, f'Optimization Failed\nfor {scenario["name"]}', 
                            transform=axes[idx].transAxes, ha='center', va='center', fontsize=14)
        
        # X-axis formatting
        axes[-1].set_xlabel('Hours', fontsize=12)
        axes[-1].set_xticks(range(0, 24, 2))
        axes[-1].set_xticklabels([f'{h}:00' for h in range(0, 24, 2)], rotation=45)
        
        plt.tight_layout()
        
        # Print detailed comparison summary
        print(f"\n" + "="*90)
        print("ALPHA SENSITIVITY ANALYSIS (QUESTION 1b)")
        print("="*90)
        
        for scenario_name, results in scenario_results.items():
            print(f"\n{scenario_name}:")
            print(f"  Alpha Value: {results['alpha_value']:.3f}")
            print(f"  Optimal Objective: {results['objective_value']:.2f}")
            print(f"    - Import Cost: {results['import_cost']:.2f} DKK")
            print(f"    - Discomfort Cost: {results['discomfort_cost']:.2f} DKK")
            print(f"  Total Discomfort: {results['total_discomfort']:.3f} kW")
            print(f"  Peak Demand Shift: {max(results['hourly_demand']) - max(results['hourly_reference']):.3f} kW")
            print(f"  Demand Variability: {max(results['hourly_demand']) - min(results['hourly_demand']):.3f} kW")
        
        # Analysis of trade-offs
        print(f"\nTRADE-OFF ANALYSIS:")
        if len(scenario_results) >= 2:
            scenarios = list(scenario_results.values())
            print(f"Objective reduction from Scenario 3 to 2: {scenarios[2]['objective_value'] - scenarios[1]['objective_value']:.2f} ")
            print(f"Discomfort increase from Scenario 3 to 2: {scenarios[1]['total_discomfort'] - scenarios[2]['total_discomfort']:.3f} kW")
        
        return fig, scenario_results


    def plot_dual_sensitivity_analysis_1c(self, figsize=(18, 12)):
        """
        Question 1c: Same as 1a analysis but with battery (P_charge, P_discharge) 
        and battery-related dual variables analysis.
        
        Create three plots showing how dual variables change with different tariff/price scenarios
        but now with battery storage capabilities.
        """
        from opt_model.opt_model import OptModel
        
        # Get data from bus_params.json (same scenarios as 1a)
        fixed_import_tariff = 0.5
        fixed_energy_price = 1.0
        
        # Dynamic energy prices from your data
        dynamic_energy_prices = [
            1.1, 1.05, 1.0, 0.9, 0.85, 1.01, 1.05, 1.2, 1.4, 1.6,
            1.5, 1.1, 1.05, 1.0, 0.95, 1.0, 1.2, 1.5, 2.1, 2.5,
            2.2, 1.8, 1.4, 1.2
        ]
        
        # Dynamic import tariff from your data (TOU Radius)
        dynamic_import_tariff = [
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.29, 0.29, 0.29, 0.29,
            0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.88, 0.88, 0.88,
            0.88, 0.29, 0.29, 0.29
        ]
        
        # Define three scenarios based on your data
        price_scenarios = [
            {
                'name': 'Scenario 1: Fixed Tariff + Fixed Energy Price + Battery',
                'description': f'Fixed import tariff ({fixed_import_tariff} DKK/kWh) + Fixed energy price ({fixed_energy_price} DKK/kWh) + Battery',
                'energy_prices': [fixed_energy_price] * 24,
                'import_tariff': [fixed_import_tariff] * 24,
                'tariff_scenario': 'import_tariff'
            },
            {
                'name': 'Scenario 2: Fixed Tariff + Dynamic Energy Price + Battery',
                'description': f'Fixed import tariff ({fixed_import_tariff} DKK/kWh) + Dynamic energy prices + Battery',
                'energy_prices': dynamic_energy_prices,
                'import_tariff': [fixed_import_tariff] * 24,
                'tariff_scenario': 'import_tariff'
            },
            {
                'name': 'Scenario 3: Dynamic Tariff + Fixed Energy Price + Battery',
                'description': f'TOU Radius tariff + Fixed energy price ({fixed_energy_price} DKK/kWh) + Battery',
                'energy_prices': [fixed_energy_price] * 24,
                'import_tariff': dynamic_import_tariff,
                'tariff_scenario': 'TOU_import_tariff_Radius'
            }
        ]
        
        # Create subplots - 3 scenarios vertically
        fig, axes = plt.subplots(6, 1, figsize=(figsize[0], figsize[1]+6), sharex=True)
        fig.suptitle('Question 1c: Battery Impact Analysis - Fixed vs Dynamic Tariffs and Energy Prices', 
                    fontsize=16, fontweight='bold')
        
        hours = list(range(24))
        scenario_results = {}
        
        for idx, scenario in enumerate(price_scenarios):
            print(f"\nRunning {scenario['name']}")
            
            # Create new model with specified scenario - CHANGED: question='1c'
            temp_model = OptModel(
                tariff_scenario=scenario['tariff_scenario'],
                question='1c',  # Question 1c for battery
                alpha_discomfort=getattr(self.model, 'alpha_discomfort', 1.0),
                consumer_type='original'
            )
            
            # Override energy prices and tariff
            temp_model.energy_prices = scenario['energy_prices']
            temp_model.active_tariff = scenario['import_tariff']
            
            # Build and run optimization
            temp_model._build_variables()
            temp_model._build_constraints()
            temp_model._build_objective_function()
            temp_model.model.update()
            temp_model.model.optimize()
            
            if temp_model.model.status == 2:  # Optimal
                temp_model._save_results()
                
                # Extract data
                hourly_demand = [temp_model.results.D_hour[t] for t in range(24)]
                hourly_import = [temp_model.results.P_imp[t] for t in range(24)]
                hourly_pv = [temp_model.results.P_PV_prod[t] for t in range(24)]
                
                # NEW: Extract battery data for Question 1c
                hourly_charge = [temp_model.results.P_charge[t] for t in range(24)]
                hourly_discharge = [temp_model.results.P_discharge[t] for t in range(24)]
                
                # Extract dual variables
                dual_values = []
                for t in range(24):
                    if t in temp_model.energy_balance_constraints:
                        dual_values.append(temp_model.energy_balance_constraints[t].Pi)
                    elif f'upper_{t}' in temp_model.energy_balance_constraints:
                        dual_values.append(temp_model.energy_balance_constraints[f'upper_{t}'].Pi)
                    else:
                        dual_values.append(0)
                
                # Calculate combined prices
                combined_prices = [ep + tariff for ep, tariff in zip(scenario['energy_prices'], scenario['import_tariff'])]
                
                # Store results
                scenario_results[scenario['name']] = {
                    'dual_values': dual_values,
                    'combined_prices': combined_prices,
                    'energy_prices': scenario['energy_prices'],
                    'import_tariff': scenario['import_tariff'],
                    'hourly_demand': hourly_demand,
                    'hourly_import': hourly_import,
                    'hourly_pv': hourly_pv,
                    'hourly_charge': hourly_charge,  # NEW
                    'hourly_discharge': hourly_discharge,  # NEW
                    'objective_value': temp_model.results.objective_value,
                    'daily_demand_dual': temp_model.daily_demand_constraint.Pi if hasattr(temp_model, 'daily_demand_constraint') else 0.0
                }
                
                # Create subplot pair for this scenario
                ax1 = axes[idx*2]      # Main power flows
                ax_battery = axes[idx*2+1]  # Battery subplot
                ax2 = ax1.twinx()      # Dual variables on main plot

                # Plot main power data on left y-axis (kW) - CLEAN without battery
                ax1.step(hours, hourly_demand, 'blue', linewidth=3, where='mid',
                        label='Hourly Demand (D_hour)', linestyle=':')
                ax1.step(hours, hourly_import, 'red', linewidth=2, where='mid',
                        label='Hourly Import', linestyle='-')
                ax1.step(hours, hourly_pv, 'yellow', linewidth=2, where='mid',
                        label='PV Production', linestyle='-')

                # Add max D_hour limit as horizontal line
                max_d_hour = getattr(temp_model, 'max_power_load', 5.0)
                ax1.axhline(y=max_d_hour, color='turquoise', linewidth=2, linestyle='--',
                        label=f'Max D_hour Limit ({max_d_hour:.1f} kW)')

                # Plot dual variables and prices on right y-axis
                ax2.step(hours, dual_values, 'black', linewidth=2, where='mid',
                        label='Energy Balance Duals', linestyle='-')
                ax2.step(hours, combined_prices, 'magenta', linewidth=2, where='mid',
                        label='Total Price (Energy + Tariff)', linestyle='--', marker='s', markersize=3)

                # Add zero line for dual variables
                ax2.axhline(y=0, color='gray', linewidth=1, linestyle=':', alpha=0.7)

                # BATTERY SUBPLOT - Much clearer visualization
                ax_battery.fill_between(hours, 0, hourly_charge, step='mid', alpha=0.8, 
                                    color='green', label='Charging', edgecolor='darkgreen', linewidth=1)
                ax_battery.fill_between(hours, 0, [-d for d in hourly_discharge], step='mid', alpha=0.8, 
                                    color='red', label='Discharging', edgecolor='darkred', linewidth=1)
                ax_battery.axhline(y=0, color='black', linewidth=1, linestyle='-')

                # Formatting for main plot
                ax1.set_ylabel('Power (kW)', fontsize=11)
                ax2.set_ylabel('Dual Variables & Prices (DKK/kWh)', fontsize=11, color='black')

                # Formatting for battery plot
                ax_battery.set_ylabel('Battery Power (kW)', fontsize=10)
                ax_battery.legend(loc='upper right', fontsize=9)
                ax_battery.grid(True, alpha=0.3)

                # Title with scenario description and cost
                ax1.set_title(f'{scenario["name"]}\nTotal Cost: {temp_model.results.objective_value:.2f} DKK', 
                            fontsize=11, fontweight='bold')

                ax1.set_xlim(-0.5, 23.5)
                ax_battery.set_xlim(-0.5, 23.5)
                ax1.grid(True, alpha=0.3)

                # Color y-axis labels
                ax1.tick_params(axis='y', labelcolor='black')
                ax2.tick_params(axis='y', labelcolor='black')

                # Legend for main plot
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
            else:
                print(f"❌ Optimization failed for {scenario['name']}")
                axes[idx].text(0.5, 0.5, f'Optimization Failed\nfor {scenario["name"]}', 
                            transform=axes[idx].transAxes, ha='center', va='center', fontsize=14)
        
        # X-axis formatting
        # X-axis formatting
        axes[-1].set_xlabel('Hours', fontsize=12)
        axes[-1].set_xticks(range(0, 24, 2))
        axes[-1].set_xticklabels([f'{h}:00' for h in range(0, 24, 2)], rotation=45)
        
        plt.tight_layout()
        
        # Print detailed comparison summary with battery analysis
        self._print_battery_scenario_analysis(scenario_results)
        
        return fig, scenario_results

    def plot_alpha_sensitivity_analysis_1c(self, figsize=(18, 12)):
        """
        Question 1c: Same as 1b analysis but with battery (P_charge, P_discharge) 
        and battery-related dual variables analysis.
        
        Analyze how alpha_discomfort affects the trade-off between cost and comfort in Question 1c with battery.
        """
        from opt_model.opt_model import OptModel
        from data_ops.data_loader import DataLoader
        
        # Load Question 1b data
        data_loader_1b = DataLoader('../data')
        data_loader_1b._load_data('question_1b', 'original')
        
        # Get energy prices and tariff from question_1b data
        energy_prices_1b = data_loader_1b.energy_prices
        import_tariff_1b = data_loader_1b.import_tariff
        
        # Calculate base alpha (equal weighting with 1b prices)
        avg_total_price = sum(energy_prices_1b[t] + import_tariff_1b for t in range(24)) / 24
        
        alpha_scenarios = [
            {
                'name': 'Scenario 1: Equal Weight (Cost ≈ Comfort) + Battery',
                'description': f'Alpha = {avg_total_price:.3f} (avg total price from 1b) + Battery flexibility',
                'alpha_value': avg_total_price
            },
            {
                'name': 'Scenario 2: Price Priority + Battery', 
                'description': f'Alpha = {avg_total_price * 0.85:.3f} - Low comfort penalty + Battery',
                'alpha_value': avg_total_price * 0.85
            },
            {
                'name': 'Scenario 3: Comfort Priority + Battery',
                'description': f'Alpha = {avg_total_price * 1.3:.3f} - High comfort penalty + Battery',
                'alpha_value': avg_total_price * 1.3
            }
        ]
        
        # Create subplots - 6 scenarios vertically
        fig, axes = plt.subplots(6, 1, figsize=(figsize[0], figsize[1]+6), sharex=True)
        fig.suptitle('Question 1c: Alpha Sensitivity Analysis with Battery - Cost vs Comfort Trade-off', 
                    fontsize=16, fontweight='bold')
        
        hours = list(range(24))
        scenario_results = {}
        
        for idx, scenario in enumerate(alpha_scenarios):
            print(f"\nRunning {scenario['name']} with alpha = {scenario['alpha_value']:.3f}")
            
            # Create new model with specified alpha - CHANGED: question='1c'
            temp_model = OptModel(
                tariff_scenario='import_tariff',  # Use flat tariff from 1b
                question='1c',  # Question 1c for battery
                alpha_discomfort=scenario['alpha_value'],
                consumer_type='original'
            )
            
            # Override with question 1b data to ensure consistency
            temp_model.energy_prices = energy_prices_1b
            temp_model.active_tariff = [import_tariff_1b] * 24  # Flat tariff
            
            # Build and run optimization
            temp_model._build_variables()
            temp_model._build_constraints()
            temp_model._build_objective_function()
            temp_model.model.update()
            temp_model.model.optimize()
            
            if temp_model.model.status == 2:  # Optimal
                temp_model._save_results()
                
                # Extract data
                hourly_demand = [temp_model.results.D_hour[t] for t in range(24)]
                hourly_import = [temp_model.results.P_imp[t] for t in range(24)]
                hourly_pv = [temp_model.results.P_PV_prod[t] for t in range(24)]
                
                # NEW: Extract battery data for Question 1c
                hourly_charge = [temp_model.results.P_charge[t] for t in range(24)]
                hourly_discharge = [temp_model.results.P_discharge[t] for t in range(24)]
                
                # Calculate reference demand for comparison
                hourly_reference = [temp_model.hourly_preference[t] * temp_model.max_power_load for t in range(24)]
                
                # Extract dual variables
                dual_values = []
                for t in range(24):
                    if t in temp_model.energy_balance_constraints:
                        dual_values.append(temp_model.energy_balance_constraints[t].Pi)
                    else:
                        dual_values.append(0)
                
                # Calculate combined prices
                combined_prices = [energy_prices_1b[t] + import_tariff_1b for t in range(24)]
                
                # Calculate discomfort metrics
                total_discomfort = temp_model.results.total_discomfort
                import_cost = sum(temp_model.results.P_imp[t] * combined_prices[t] for t in range(24))
                discomfort_cost = scenario['alpha_value'] * total_discomfort
                
                # Store results
                scenario_results[scenario['name']] = {
                    'alpha_value': scenario['alpha_value'],
                    'dual_values': dual_values,
                    'combined_prices': combined_prices,
                    'hourly_demand': hourly_demand,
                    'hourly_reference': hourly_reference,
                    'hourly_import': hourly_import,
                    'hourly_pv': hourly_pv,
                    'hourly_charge': hourly_charge,  # NEW
                    'hourly_discharge': hourly_discharge,  # NEW
                    'objective_value': temp_model.results.objective_value,
                    'total_discomfort': total_discomfort,
                    'import_cost': import_cost,
                    'discomfort_cost': discomfort_cost
                }
                
                                # Create subplot pair for this scenario
                ax1 = axes[idx*2]      # Main power flows
                ax_battery = axes[idx*2+1]  # Battery subplot
                ax2 = ax1.twinx()      # Dual variables on main plot

                # Plot main power data on left y-axis (kW) - CLEAN without battery
                ax1.step(hours, hourly_demand, 'blue', linewidth=3, where='mid',
                        label='Actual Demand (D_hour)', linestyle='-')
                ax1.step(hours, hourly_reference, 'steelblue', linewidth=2, where='mid',
                        label='Reference Demand', linestyle=':', alpha=0.7)
                ax1.step(hours, hourly_import, 'red', linewidth=2, where='mid',
                        label='Hourly Import', linestyle='-')
                ax1.step(hours, hourly_pv, 'yellow', linewidth=2, where='mid',
                        label='PV Production', linestyle='-')

                # Add max D_hour limit as horizontal line
                max_d_hour = getattr(temp_model, 'max_power_load', 5.0)
                ax1.axhline(y=max_d_hour, color='turquoise', linewidth=2, linestyle='--',
                        label=f'Max D_hour Limit ({max_d_hour:.1f} kW)')

                # Plot dual variables and prices on right y-axis
                ax2.step(hours, dual_values, 'black', linewidth=2, where='mid',
                        label='Energy Balance Duals', linestyle='-')
                ax2.step(hours, combined_prices, 'magenta', linewidth=2, where='mid',
                        label='Total Price (Energy + Tariff)', linestyle='--', marker='s', markersize=3)

                # Add zero line for dual variables
                ax2.axhline(y=0, color='gray', linewidth=1, linestyle=':', alpha=0.7)

                # BATTERY SUBPLOT - Much clearer visualization
                ax_battery.fill_between(hours, 0, hourly_charge, step='mid', alpha=0.8, 
                                    color='green', label='Charging', edgecolor='darkgreen', linewidth=1)
                ax_battery.fill_between(hours, 0, [-d for d in hourly_discharge], step='mid', alpha=0.8, 
                                    color='red', label='Discharging', edgecolor='darkred', linewidth=1)
                ax_battery.axhline(y=0, color='black', linewidth=1, linestyle='-')

                # Formatting for main plot
                ax1.set_ylabel('Power (kW)', fontsize=11)
                ax2.set_ylabel('Dual Variables & Prices (DKK/kWh)', fontsize=11, color='black')

                # Formatting for battery plot
                ax_battery.set_ylabel('Battery Power (kW)', fontsize=10)
                ax_battery.legend(loc='upper right', fontsize=9)
                ax_battery.grid(True, alpha=0.3)

                # Title with scenario description and costs breakdown
                ax1.set_title(f'{scenario["name"]}\n'
                            f'Optimal Objective: {temp_model.results.objective_value:.2f} DKK '
                            f'(Import: {import_cost:.2f} + Discomfort: {discomfort_cost:.2f})', 
                            fontsize=11, fontweight='bold')

                ax1.set_xlim(-0.5, 23.5)
                ax_battery.set_xlim(-0.5, 23.5)
                ax1.grid(True, alpha=0.3)

                # Color y-axis labels
                ax1.tick_params(axis='y', labelcolor='black')
                ax2.tick_params(axis='y', labelcolor='black')

                # Legend for main plot
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
            else:
                print(f"❌ Optimization failed for {scenario['name']}")
                axes[idx].text(0.5, 0.5, f'Optimization Failed\nfor {scenario["name"]}', 
                            transform=axes[idx].transAxes, ha='center', va='center', fontsize=14)
        
        # X-axis formatting (bottom subplot which is now a battery plot)
        axes[-1].set_xlabel('Hours', fontsize=12)
        axes[-1].set_xticks(range(0, 24, 2))
        axes[-1].set_xticklabels([f'{h}:00' for h in range(0, 24, 2)], rotation=45)
        plt.tight_layout()
        
        # Print detailed comparison summary with battery analysis
        self._print_battery_alpha_analysis(scenario_results)
        
        return fig, scenario_results

    def _print_battery_scenario_analysis(self, scenario_results):
        """Print battery-specific analysis for tariff/price scenarios"""
        print(f"\n" + "="*90)
        print("QUESTION 1c: BATTERY IMPACT ANALYSIS - TARIFF/PRICE SCENARIOS")
        print("="*90)
        
        for scenario_name, results in scenario_results.items():
            print(f"\n{scenario_name}:")
            print(f"  Total Cost: {results['objective_value']:.2f} DKK")
            print(f"  Total Import: {sum(results['hourly_import']):.2f} kW")
            print(f"  Total Battery Charging: {sum(results['hourly_charge']):.2f} kW")
            print(f"  Total Battery Discharging: {sum(results['hourly_discharge']):.2f} kW")
            
            if sum(results['hourly_charge']) > 0:
                efficiency = (sum(results['hourly_discharge']) / sum(results['hourly_charge'])) * 100
                print(f"  Battery Round-trip Efficiency: {efficiency:.1f}%")
            
            print(f"  Peak Import Hour: {results['hourly_import'].index(max(results['hourly_import']))} (Import: {max(results['hourly_import']):.2f} kW)")
            print(f"  Daily Demand Constraint Dual: {results['daily_demand_dual']:.4f}")
            
            # Find significant charging/discharging hours
            charge_hours = [t for t, charge in enumerate(results['hourly_charge']) if charge > 0.01]
            discharge_hours = [t for t, discharge in enumerate(results['hourly_discharge']) if discharge > 0.01]
            
            if charge_hours:
                print(f"  Battery Charging Hours: {charge_hours}")
            if discharge_hours:
                print(f"  Battery Discharging Hours: {discharge_hours}")

    def _print_battery_alpha_analysis(self, scenario_results):
        """Print battery-specific analysis for alpha scenarios"""
        print(f"\n" + "="*90)
        print("QUESTION 1c: BATTERY IMPACT ANALYSIS - ALPHA SENSITIVITY")
        print("="*90)
        
        for scenario_name, results in scenario_results.items():
            print(f"\n{scenario_name}:")
            print(f"  Alpha Value: {results['alpha_value']:.3f}")
            print(f"  Optimal Objective: {results['objective_value']:.2f} DKK")
            print(f"    - Import Cost: {results['import_cost']:.2f} DKK")
            print(f"    - Discomfort Cost: {results['discomfort_cost']:.2f} DKK")
            print(f"  Total Discomfort: {results['total_discomfort']:.3f} kW")
            print(f"  Total Battery Charging: {sum(results['hourly_charge']):.2f} kW")
            print(f"  Total Battery Discharging: {sum(results['hourly_discharge']):.2f} kW")
            
            if sum(results['hourly_charge']) > 0:
                efficiency = (sum(results['hourly_discharge']) / sum(results['hourly_charge'])) * 100
                print(f"  Battery Round-trip Efficiency: {efficiency:.1f}%")
            
            print(f"  Peak Demand Shift: {max(results['hourly_demand']) - max(results['hourly_reference']):.3f} kW")

    def plot_question_1a_results(self, figsize=(12, 8)):
        """
        Plot Question 1a results with 4 elements:
        - Chosen tariff scenario
        - Hourly import
        - Energy prices  
        - Hourly load/demand
        
        Args:
            figsize: Figure size tuple (width, height)
        """
        # Create figure and axes
        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = ax1.twinx()  # Second y-axis
        
        # Hours for x-axis
        hours = list(range(24))
        
        # Get data from model
        tariff_values = self.model.active_tariff
        energy_prices = self.model.energy_prices
        
        # Get hourly import from optimization results
        if hasattr(self.model, 'results') and hasattr(self.model.results, 'P_imp'):
            hourly_import = [self.model.results.P_imp[t] for t in range(24)]
        else:
            # If no results yet, use zeros or run optimization
            hourly_import = [0] * 24
            
        # Get hourly demand/load from D_hour variable
        if hasattr(self.model, 'results') and hasattr(self.model.results, 'D_hour'):
            hourly_demand = [self.model.results.D_hour[t] for t in range(24)]
        elif hasattr(self.model, 'D_hour'):
            # If D_hour exists as model attribute
            hourly_demand = [self.model.D_hour[t].X for t in range(24)]
        else:
            # Fallback if no D_hour available
            print("Warning: D_hour not found, using default demand values")
            hourly_demand = [0] * 24

        # Get PV production from optimization results
        if hasattr(self.model, 'results') and hasattr(self.model.results, 'P_PV_prod'):
            pv_production = [self.model.results.P_PV_prod[t] for t in range(24)]
        elif hasattr(self.model, 'P_PV_prod'):
            # If P_PV_prod exists as model attribute
            pv_production = [self.model.P_PV_prod[t].X for t in range(24)]
        else:
            # Fallback: calculate from PV capacity and hourly ratio
            if hasattr(self.model, 'pv_capacity') and hasattr(self.model, 'pv_hourly_ratio'):
                pv_production = [self.model.pv_capacity * self.model.pv_hourly_ratio[t] for t in range(24)]
            else:
                print("Warning: PV production not found, using zeros")
                pv_production = [0] * 24
            
    # Plot power data on left y-axis (kW) as step diagrams
        line1 = ax1.step(hours, hourly_import, 'b-', linewidth=2, where='mid',
                        label='Hourly Import')
        line2 = ax1.step(hours, hourly_demand, 'g-', linewidth=2, where='mid',
                        label='Hourly Load/Demand')
        line3_pv = ax1.step(hours, pv_production, 'orange', linewidth=2, where='mid',
                        label='PV Production')
        line4_demand = ax1.step(hours, hourly_demand, 'lime', linewidth=2, where='mid',
                       label='Hourly Demand (D_hour)', linestyle='--')

        
        # Plot price data on right y-axis (DKK/kWh) as step diagrams
        line4 = ax2.step(hours, tariff_values, 'r-', linewidth=2, where='mid',
                        label='Tariff Scenario')
        line5 = ax2.step(hours, energy_prices, 'm--', linewidth=2, where='mid',
                        label='Energy Prices')
        
        # Add markers at each hour for better readability
        ax1.scatter(hours, hourly_import, color='blue', s=30, zorder=5)
        ax1.scatter(hours, hourly_demand, color='green', s=30, zorder=5)
        ax1.scatter(hours, pv_production, color='orange', s=30, zorder=5)
        ax1.scatter(hours, hourly_demand, color='lime', s=30, zorder=5)
        ax2.scatter(hours, tariff_values, color='red', s=30, zorder=5)
        ax2.scatter(hours, energy_prices, color='magenta', s=30, zorder=5)
        
        # Set x-axis
        ax1.set_xlim(-0.5, 23.5)
        ax1.set_xticks(range(0, 24, 2))
        ax1.set_xticklabels([f'{h}:00' for h in range(0, 24, 2)], rotation=45)
                
        # Grid
        ax1.grid(True, alpha=0.3)
        
        # Color y-axis labels
        ax1.tick_params(axis='y', labelcolor='black')
        ax2.tick_params(axis='y', labelcolor='black')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
        
        # Title
        title = f'Question 1a Results - Tariff Scenario: {self._get_tariff_name()}'
        plt.title(title, fontsize=14, fontweight='bold')
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    
    
    def _get_tariff_name(self):
        """Get the name of the current tariff scenario"""
        # Compare active tariff with known tariffs to identify which one is used
        if hasattr(self.model, 'TOU_import_tariff_Radius') and self.model.active_tariff == self.model.TOU_import_tariff_Radius:
            return 'TOU Radius'
        elif hasattr(self.model, 'TOU_import_tariff_N1') and self.model.active_tariff == self.model.TOU_import_tariff_N1:
            return 'TOU N1'
        elif hasattr(self.model, 'TOU_import_tariff_Bornholm') and self.model.active_tariff == self.model.TOU_import_tariff_Bornholm:
            return 'TOU Bornholm'
        elif hasattr(self.model, 'import_tariff') and all(t == self.model.import_tariff for t in self.model.active_tariff):
            return 'Flat Import Tariff'
        else:
            return 'Unknown Tariff'
    
    def save_plot(self, fig, filename, dpi=300):
        """
        Save the plot to file
        
        Args:
            fig: matplotlib figure
            filename: output filename
            dpi: resolution for saved figure
        """
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to: {filename}")
    
    def show_plot(self):
        """Display the plot"""
        plt.show()
    

    
    def plot_optimized_stacked_comparison(self, figsize=(15, 10)):
        """
        Create stacked plots with actual optimization results for each tariff scenario.
        This runs optimization for each scenario to get real import values.
        """
        from opt_model.opt_model import OptModel
        
        # Define tariff scenarios to compare
        scenarios = [
            ('Flat Import Tariff', 'import_tariff'),
            ('TOU Radius', 'TOU_import_tariff_Radius'), 
            ('TOU Trefor Bornholm', 'TOU_import_tariff_Bornholm')  # Changed from TOU_bornholm
        ]
        
        # Create subplots
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        fig.suptitle('Optimized Stacked Comparison: Three Tariff Scenarios', fontsize=16, fontweight='bold')
        
        hours = list(range(24))
        
        for idx, (scenario_name, tariff_key) in enumerate(scenarios):
            print(f"Running optimization for {scenario_name}...")
            
            # Create new model for this scenario
            temp_model = OptModel(
                tariff_scenario=tariff_key,
                question=self.model.question,
                alpha_discomfort=getattr(self.model, 'alpha_discomfort', 1.0),
                consumer_type='original'
            )
            
            # Run optimization
            temp_model.run()
            
            # Extract results
            pv_production = [temp_model.results.P_PV_prod[t] for t in range(24)]
            hourly_import = [temp_model.results.P_imp[t] for t in range(24)]
            energy_prices = temp_model.energy_prices
            tariff_values = temp_model.active_tariff
            hourly_demand = [temp_model.results.D_hour[t] for t in range(24)]
            
            # Create subplot
            ax1 = axes[idx]
            ax2 = ax1.twinx()
            
            # Stacked areas for power (kW)
            ax1.fill_between(hours, 0, pv_production, 
                            step='mid', alpha=0.7, color='orange', label='PV Production')
            ax1.fill_between(hours, pv_production, 
                            [pv + imp for pv, imp in zip(pv_production, hourly_import)],
                            step='mid', alpha=0.7, color='blue', label='Grid Import')
            
            # Stacked areas for prices (DKK/kWh)  
            ax2.fill_between(hours, 0, energy_prices,
                            step='mid', alpha=0.5, color='purple', label='Energy Prices')
            ax2.fill_between(hours, energy_prices,
                            [ep + tariff for ep, tariff in zip(energy_prices, tariff_values)],
                            step='mid', alpha=0.5, color='red', label='Tariff')
            ax1.step(hours, hourly_demand, 'lime', linewidth=2, where='mid',
                    label='Demand (D_hour)', linestyle='--')
            
            # Formatting
            ax1.set_ylabel('Power (kW)', fontsize=10)
            ax2.set_ylabel('Price (DKK/kWh)', fontsize=10)

            # Calculate total cost manually since total_cost attribute doesn't exist
            try:
                if hasattr(temp_model.results, 'total_cost'):
                    total_cost = temp_model.results.total_cost
                else:
                    # Calculate cost manually from results
                    import_costs = sum(temp_model.results.P_imp[t] * temp_model.active_tariff[t] 
                                    for t in range(24))
                    energy_costs = sum(temp_model.results.P_imp[t] * temp_model.energy_prices[t] 
                                    for t in range(24))
                    total_cost = import_costs + energy_costs
                
                ax1.set_title(f'{scenario_name} - Total Cost: {total_cost:.2f} DKK', 
                            fontsize=12, fontweight='bold')
            except:
                # Fallback if cost calculation fails
                ax1.set_title(f'{scenario_name}', fontsize=12, fontweight='bold')

            ax1.set_xlim(-0.5, 23.5)
            ax1.grid(True, alpha=0.3)
            
            # Legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
        
        # X-axis formatting
        axes[-1].set_xlabel('Hours', fontsize=12)
        axes[-1].set_xticks(range(0, 24, 2))
        axes[-1].set_xticklabels([f'{h}:00' for h in range(0, 24, 2)], rotation=45)
        
        plt.tight_layout()
        return fig

    def conduct_battery_profitability_experiment(self, battery_cost_per_kwh=3000):
        """
        Quantify battery profitability across consumer preferences and cost structures.
        
        Returns comprehensive profitability metrics without plotting.
        """
        import pandas as pd
        import numpy as np
        from opt_model.opt_model import OptModel
        
        # Define scenarios based on your 1a-1c findings
        market_scenarios = [
            {
                'name': 'Low Volatility (Fixed-Fixed)',
                'energy_prices': [1.0] * 24,
                'tariff_scenario': 'import_tariff',  # Flat 0.5
                'volatility_index': 0.0
            },
            {
                'name': 'Medium Volatility (Fixed-Dynamic)',
                'energy_prices': [1.1, 1.05, 1.0, 0.9, 0.85, 1.01, 1.05, 1.2, 1.4, 1.6,
                                1.5, 1.1, 1.05, 1.0, 0.95, 1.0, 1.2, 1.5, 2.1, 2.5,
                                2.2, 1.8, 1.4, 1.2],
                'tariff_scenario': 'import_tariff',  # Flat 0.5
                'volatility_index': None  # Calculate later
            },
            {
                'name': 'High Volatility (TOU-Fixed)',
                'energy_prices': [1.0] * 24,
                'tariff_scenario': 'TOU_import_tariff_Radius',
                'volatility_index': None  # Calculate later
            },
            {
                'name': 'Ultra Volatility (TOU-Dynamic)',
                'energy_prices': [1.1, 1.05, 1.0, 0.9, 0.85, 1.01, 1.05, 1.2, 1.4, 1.6,
                                1.5, 1.1, 1.05, 1.0, 0.95, 1.0, 1.2, 1.5, 2.1, 2.5,
                                2.2, 1.8, 1.4, 1.2],
                'tariff_scenario': 'TOU_import_tariff_Radius',
                'volatility_index': None  # Calculate later
            }
        ]
        
        # Consumer preference scenarios (based on 1b analysis)
        avg_price = 1.5
        consumer_scenarios = [
            {
                'name': 'Rigid Consumer',
                'alpha_discomfort': avg_price * 2.0,
                'flexibility_index': 'Low'
            },
            {
                'name': 'Balanced Consumer', 
                'alpha_discomfort': avg_price * 1.0,
                'flexibility_index': 'Medium'
            },
            {
                'name': 'Flexible Consumer',
                'alpha_discomfort': avg_price * 0.5,
                'flexibility_index': 'High'
            }
        ]
        
        # Battery investment scenarios
        battery_scalars = np.linspace(0, 2.0, 21)  # 0% to 200% in 10% increments
        
        results = []
        
        print("="*80)
        print("BATTERY PROFITABILITY ANALYSIS")
        print("="*80)
        
        for market in market_scenarios:
            for consumer in consumer_scenarios:
                print(f"\nAnalyzing: {market['name']} × {consumer['name']}")
                
                # Calculate baseline (no battery) cost
                baseline_cost = self._calculate_baseline_cost(market, consumer)
                
                profitability_data = []
                
                for battery_scalar in battery_scalars:
                    if battery_scalar == 0:
                        # No battery case
                        net_benefit = 0
                        payback_period = float('inf')
                        roi = 0
                    else:
                        # With battery case - use Question 2b optimization
                        net_benefit, payback_period, roi, operational_metrics = self._calculate_battery_economics(
                            market, consumer, battery_scalar, battery_cost_per_kwh, baseline_cost
                        )
                    
                    profitability_data.append({
                        'battery_scalar': battery_scalar,
                        'net_annual_benefit': net_benefit,
                        'payback_period': payback_period,
                        'roi_percent': roi,
                        'is_profitable': net_benefit > 0
                    })
                
                # Find optimal battery size for this scenario
                optimal_result = max(profitability_data, key=lambda x: x['net_annual_benefit'])
                
                # Calculate break-even battery cost
                break_even_cost = self._calculate_break_even_cost(market, consumer, baseline_cost)
                
                results.append({
                    'market_scenario': market['name'],
                    'consumer_type': consumer['name'],
                    'consumer_alpha': consumer['alpha_discomfort'],
                    'baseline_annual_cost': baseline_cost * 365,  # Annualize
                    'optimal_battery_scalar': optimal_result['battery_scalar'],
                    'optimal_annual_benefit': optimal_result['net_annual_benefit'],
                    'optimal_roi': optimal_result['roi_percent'],
                    'optimal_payback': optimal_result['payback_period'],
                    'break_even_battery_cost': break_even_cost,
                    'profitability_threshold': optimal_result['battery_scalar'] if optimal_result['is_profitable'] else None,
                    'detailed_profitability': profitability_data
                })
        
        # Generate comprehensive summary
        self._generate_profitability_summary(results, battery_cost_per_kwh)
        
        return results

    def _calculate_baseline_cost(self, market, consumer):
        """Calculate daily cost without battery (Question 1b)"""
        from opt_model.opt_model import OptModel
        
        model_no_battery = OptModel(
            tariff_scenario=market['tariff_scenario'],
            question='1b',
            alpha_discomfort=consumer['alpha_discomfort'],
            consumer_type='original'
        )
        
        # IMPORTANT: Override prices AFTER model creation but BEFORE building
        if 'energy_prices' in market:
            model_no_battery.energy_prices = market['energy_prices']
            print(f"DEBUG: Setting energy prices to {market['energy_prices'][:4]}...")  # Debug print
        
        # Make sure to rebuild the model with new prices
        model_no_battery._build_variables()
        model_no_battery._build_constraints()
        model_no_battery._build_objective_function()
        model_no_battery.model.update()
        model_no_battery.model.optimize()
        
        # ADD THIS: Save results after optimization
        if model_no_battery.model.status == 2:  # Optimal
            model_no_battery._save_results()
            return model_no_battery.results.objective_value
        else:
            print(f"WARNING: Optimization failed for baseline cost calculation")
            return float('inf')  # Return high cost if optimization fails


    def _calculate_battery_economics(self, market, consumer, battery_scalar, cost_per_kwh, baseline_cost):
        """Calculate battery economics using Question 2b optimization"""
        from opt_model.opt_model import OptModel
        
        # Create Question 2b model (investment optimization)
        model_with_battery = OptModel(
            tariff_scenario=market['tariff_scenario'],
            question='2b',
            alpha_discomfort=consumer['alpha_discomfort'],
            consumer_type='original'
        )
        
        # Override prices if needed
        if 'energy_prices' in market:
            model_with_battery.energy_prices = market['energy_prices']
            print(f"DEBUG: Setting energy prices for battery model to {market['energy_prices'][:4]}...")
        
        # Build the model first
        model_with_battery._build_variables()
        model_with_battery._build_constraints()
        model_with_battery._build_objective_function()
        
        # Fix battery scalar to specific value for this analysis
        model_with_battery.battery_scalar.lb = battery_scalar
        model_with_battery.battery_scalar.ub = battery_scalar
        
        model_with_battery.model.update()
        model_with_battery.model.optimize()
        
        if model_with_battery.model.status == 2:  # Optimal
            # ADD THIS: Save results after optimization
            model_with_battery._save_results()
            
            # Extract results
            daily_operational_cost = model_with_battery.results.objective_value
            
            # For Question 2b, check if battery_capital_cost exists in results
            if hasattr(model_with_battery.results, 'battery_capital_cost'):
                daily_capital_cost = model_with_battery.results.battery_capital_cost * 30000 / (10*365)
            else:
                # Calculate capital cost manually
                battery_capacity = battery_scalar * getattr(model_with_battery, 'storage_capacity', 10.0)  # Default 10 kWh
                total_capital_cost = battery_capacity * cost_per_kwh
                daily_capital_cost = total_capital_cost / (10 * 365)  # 10 years, daily
            
            # Calculate economics
            daily_operational_savings = baseline_cost - (daily_operational_cost - daily_capital_cost)
            annual_operational_savings = daily_operational_savings * 365
            
            total_capital_cost = battery_scalar * getattr(model_with_battery, 'storage_capacity', 10.0) * cost_per_kwh
            net_annual_benefit = annual_operational_savings - (total_capital_cost / 10)  # 10-year life
            
            # Calculate payback and ROI
            if annual_operational_savings > 0:
                payback_period = total_capital_cost / annual_operational_savings
                roi = (annual_operational_savings / total_capital_cost) * 100
            else:
                payback_period = float('inf')
                roi = -100  # Loss
            
            # Operational metrics
            operational_metrics = {
                'battery_utilization': sum(getattr(model_with_battery.results, 'P_charge', [0]*24)[t] + 
                                        getattr(model_with_battery.results, 'P_discharge', [0]*24)[t] 
                                        for t in range(24)),
                'peak_shaving': baseline_cost - daily_operational_cost,
                'cycling_frequency': sum(getattr(model_with_battery.results, 'P_charge', [0]*24)[t] 
                                    for t in range(24)) / (battery_scalar * getattr(model_with_battery, 'storage_capacity', 10.0)) if battery_scalar > 0 else 0
            }
            
            return net_annual_benefit, payback_period, roi, operational_metrics
        else:
            print(f"WARNING: Battery optimization failed for scalar {battery_scalar}")
            return -float('inf'), float('inf'), -100, {}

    def _calculate_break_even_cost(self, market, consumer, baseline_cost):
        """Calculate break-even battery cost per kWh"""
        # Use binary search or simple iteration to find break-even cost
        for cost_per_kwh in range(1000, 10000, 500):  # Test from 1000 to 10000 DKK/kWh
            net_benefit, _, _, _ = self._calculate_battery_economics(
                market, consumer, 1.0, cost_per_kwh, baseline_cost  # Test with 100% battery
            )
            if net_benefit <= 0:
                return cost_per_kwh - 500  # Previous cost was break-even
        return 10000  # High value if always profitable

    def _generate_profitability_summary(self, results, battery_cost_per_kwh):
        """Generate comprehensive profitability summary"""
        print(f"\n" + "="*80)
        print("PROFITABILITY SUMMARY")
        print("="*80)
        print(f"Battery Cost Assumption: {battery_cost_per_kwh:,} DKK/kWh")
        print(f"Investment Horizon: 10 years")
        
        # Convert to DataFrame for analysis
        import pandas as pd
        df = pd.DataFrame(results)
        
        print(f"\n" + "-"*60)
        print("OPTIMAL INVESTMENT RECOMMENDATIONS")
        print("-"*60)
        
        for _, row in df.iterrows():
            print(f"\n{row['market_scenario']} × {row['consumer_type']}:")
            if row['optimal_annual_benefit'] > 0:
                print(f"  ✅ PROFITABLE")
                print(f"  Optimal Battery Size: {row['optimal_battery_scalar']:.1f}× base capacity")
                print(f"  Annual Net Benefit: {row['optimal_annual_benefit']:,.0f} DKK/year")
                print(f"  ROI: {row['optimal_roi']:.1f}%")
                print(f"  Payback Period: {row['optimal_payback']:.1f} years")
            else:
                print(f"  ❌ NOT PROFITABLE")
                print(f"  Break-even Battery Cost: {row['break_even_battery_cost']:,.0f} DKK/kWh")
        
        # Market volatility insights
        print(f"\n" + "-"*60)
        print("MARKET VOLATILITY INSIGHTS")
        print("-"*60)
        
        market_summary = df.groupby('market_scenario').agg({
            'optimal_annual_benefit': 'mean',
            'optimal_battery_scalar': 'mean'
        }).round(2)
        
        for market, data in market_summary.iterrows():
            print(f"{market}: Avg Benefit = {data['optimal_annual_benefit']:,.0f} DKK/year, "
                f"Avg Optimal Size = {data['optimal_battery_scalar']:.1f}×")
        
        # Consumer type insights
        print(f"\n" + "-"*60)
        print("CONSUMER TYPE INSIGHTS")
        print("-"*60)
        
        consumer_summary = df.groupby('consumer_type').agg({
            'optimal_annual_benefit': 'mean',
            'optimal_battery_scalar': 'mean'
        }).round(2)
        
        for consumer, data in consumer_summary.iterrows():
            print(f"{consumer}: Avg Benefit = {data['optimal_annual_benefit']:,.0f} DKK/year, "
                f"Avg Optimal Size = {data['optimal_battery_scalar']:.1f}×")
