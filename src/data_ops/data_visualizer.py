import json
import csv
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys

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

