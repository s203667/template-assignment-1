import json
import csv
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys

# Add path setup
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

from opt_model.opt_model import OptModel

class DataVisualizer:
    """Visualizer for optimization results across different tariff scenarios."""

    def __init__(self):
        self.scenarios = {
            'Flat Rate': 'import_tariff',
            'Radius TOU': 'TOU_import_tariff_Radius',
            'N1 TOU': 'TOU_import_tariff_N1',
            'Bornholm TOU': 'TOU_import_tariff_Bornholm'
        }
        self.results = {}

    def run_scenario_analysis(self):
        """Run optimization for each tariff scenario"""
        print("Running tariff scenario analysis...")

        for scenario_name, tariff_key in self.scenarios.items():
            print(f"  - {scenario_name}")

            # Create and run model for this scenario
            model = OptModel(tariff_scenario=tariff_key, question='1a', alpha_discomfort=2, consumer_type='original')
            model.run()

            # Store key results
            self.results[scenario_name] = {
                'total_cost': model.results.objective_value,
                'hourly_demand': list(model.results.D_hour.values()),
                'hourly_import': list(model.results.P_imp.values()),
                'hourly_export': list(model.results.P_exp.values()),
                'total_demand': sum(model.results.D_hour.values()),
                'active_tariff': model.active_tariff
            }

        print("✓ Analysis complete")

    def plot_cost_comparison(self):
        """Simple cost comparison bar chart"""
        if not self.results:
            print("No results to plot. Run scenario analysis first.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        scenarios = list(self.results.keys())
        costs = [self.results[scenario]['total_cost'] for scenario in scenarios]

        # Total costs
        bars = ax1.bar(scenarios, costs, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
        ax1.set_title('Daily Costs by Tariff Structure', fontweight='bold')
        ax1.set_ylabel('Cost (DKK)')
        ax1.tick_params(axis='x', rotation=45)

        # Add values on bars
        for bar, cost in zip(bars, costs):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{cost:.1f}', ha='center', va='bottom', fontweight='bold')

        # Savings vs flat rate
        flat_cost = costs[0]  # First scenario is flat rate
        savings_pct = [(flat_cost - cost)/flat_cost * 100 for cost in costs]

        bars2 = ax2.bar(scenarios, savings_pct, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
        ax2.set_title('Savings vs Flat Rate', fontweight='bold')
        ax2.set_ylabel('Savings (%)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        # Add values on bars
        for bar, pct in zip(bars2, savings_pct):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.3 if height >= 0 else height - 0.3,
                    f'{pct:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')

        plt.tight_layout()
        plt.show()

    def plot_demand_patterns(self):
        """Plot hourly demand patterns for all scenarios"""
        if not self.results:
            print("No results to plot. Run scenario analysis first.")
            return

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        hours = range(24)
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

        for i, (scenario, data) in enumerate(self.results.items()):
            ax.plot(hours, data['hourly_demand'],
                   linewidth=2.5, label=scenario, color=colors[i], marker='o', markersize=4)

        ax.set_title('Hourly Demand Patterns by Tariff Structure', fontsize=14, fontweight='bold')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Demand (kW)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 23)

        plt.tight_layout()
        plt.show()

    def plot_tariff_vs_demand(self):
        """Plot tariff rates vs demand response for each scenario"""
        if not self.results:
            print("No results to plot. Run scenario analysis first.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        hours = range(24)

        for idx, (scenario, data) in enumerate(self.results.items()):
            ax = axes[idx]

            # Create twin axis for demand
            ax2 = ax.twinx()

            # Plot tariff rates
            if scenario == 'Flat Rate':
                tariff_rates = [0.5] * 24  # Flat rate
            else:
                tariff_rates = data['active_tariff']

            line1 = ax.plot(hours, tariff_rates, 'r-', linewidth=2, label='Tariff Rate', marker='s', markersize=4)
            line2 = ax2.plot(hours, data['hourly_demand'], 'b-', linewidth=2, label='Demand', marker='o', markersize=4)

            ax.set_xlabel('Hour of Day')
            ax.set_ylabel('Tariff Rate (DKK/kWh)', color='r')
            ax2.set_ylabel('Demand (kW)', color='b')
            ax.set_title(f'{scenario}', fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left')

        plt.tight_layout()
        plt.show()

    def plot_flexibility_metrics(self):
        """Calculate and plot flexibility metrics"""
        if not self.results:
            print("No results to plot. Run scenario analysis first.")
            return

        # Calculate flexibility metrics
        flexibility_data = {}
        for scenario, data in self.results.items():
            demands = data['hourly_demand']
            flexibility_data[scenario] = {
                'peak_demand': max(demands),
                'min_demand': min(demands),
                'demand_range': max(demands) - min(demands),
                'std_deviation': np.std(demands)
            }

        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        scenarios = list(flexibility_data.keys())
        ranges = [flexibility_data[s]['demand_range'] for s in scenarios]
        std_devs = [flexibility_data[s]['std_deviation'] for s in scenarios]

        # Demand range (flexibility indicator)
        bars1 = ax1.bar(scenarios, ranges, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
        ax1.set_title('Demand Flexibility\n(Peak - Min Demand)', fontweight='bold')
        ax1.set_ylabel('Demand Range (kW)')
        ax1.tick_params(axis='x', rotation=45)

        # Add values on bars
        for bar, value in zip(bars1, ranges):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

        # Standard deviation (variability)
        bars2 = ax2.bar(scenarios, std_devs, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
        ax2.set_title('Demand Variability\n(Standard Deviation)', fontweight='bold')
        ax2.set_ylabel('Std Deviation (kW)')
        ax2.tick_params(axis='x', rotation=45)

        # Add values on bars
        for bar, value in zip(bars2, std_devs):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.show()

    def generate_summary_table(self):
        """Generate a summary table of results"""
        if not self.results:
            print("No results to summarize. Run scenario analysis first.")
            return

        summary_data = []
        flat_cost = self.results['Flat Rate']['total_cost']

        for scenario, data in self.results.items():
            cost_savings = flat_cost - data['total_cost']
            savings_pct = (cost_savings / flat_cost) * 100 if flat_cost > 0 else 0

            # Calculate flexibility metrics
            demands = data['hourly_demand']
            demand_range = max(demands) - min(demands)

            summary_data.append({
                'Scenario': scenario,
                'Total Cost (DKK)': f"{data['total_cost']:.2f}",
                'Cost Savings (DKK)': f"{cost_savings:.2f}",
                'Savings (%)': f"{savings_pct:.1f}%",
                'Peak Demand (kW)': f"{max(demands):.2f}",
                'Min Demand (kW)': f"{min(demands):.2f}",
                'Demand Range (kW)': f"{demand_range:.2f}",
                'Total Energy (kWh)': f"{data['total_demand']:.2f}"
            })

        df = pd.DataFrame(summary_data)
        print("\n" + "="*100)
        print("TARIFF SCENARIO COMPARISON SUMMARY")
        print("="*100)
        print(df.to_string(index=False))
        print("="*100)

        return df

    def run_complete_analysis(self):
        """Run complete analysis and generate all visualizations"""
        print("Starting complete tariff scenario analysis...\n")

        # Run scenarios
        self.run_scenario_analysis()

        # Generate visualizations
        print("\nGenerating visualizations...")
        self.plot_cost_comparison()
        self.plot_demand_patterns()
        self.plot_tariff_vs_demand()
        self.plot_flexibility_metrics()

        # Generate summary
        print("\nGenerating summary table...")
        self.generate_summary_table()

        print("\n✓ Complete analysis finished!")
        print("\nKey Insights:")
        print("- Time-of-use tariffs incentivize demand shifting to low-cost periods")
        print("- TOU tariffs increase consumer flexibility (higher demand range)")
        print("- Cost savings depend on tariff structure and price differentials")
        print("- Bornholm TOU offers lowest costs due to lower rate structure")
