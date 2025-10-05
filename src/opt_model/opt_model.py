from pathlib import Path
import numpy as np
import pandas as pd
import gurobipy as gp
import xarray as xr


import sys
import os
from pathlib import Path

current_dir = Path(__file__).parent  # opt_model directory
src_dir = current_dir.parent         # src directory
sys.path.insert(0, str(src_dir))

from data_ops.data_loader import DataLoader
from gurobipy import GRB

#data_loader = DataLoader('../data')  # Changed from '../data' to '../../data'

class Expando(object):
    '''
        A small class which can have attributes set
    '''
    pass


class OptModel():

    def __init__(self, tariff_scenario: str, question: str, alpha_discomfort: float, consumer_type: str): # initialize class
        #self.data = input_data # define data attributes
        self.results = Expando() # define results attributes
        self.question = question # question 1a or 1b or 1c
        self.alpha_discomfort = alpha_discomfort # weight of discomfort in objective function
        self.consumer_type = consumer_type # type of consumer (original, flexible, etc.)

        # Load data once and store as instance attributes
        self.T = 24

        # Create data loader and load appropriate question data
        data_loader = DataLoader('../data')  # Point to data folder
    
        if question == '1a':
            data_loader._load_data('question_1a', consumer_type)  # Load from question_1a folder
        elif question == '1b':  # question == '1b'
            data_loader._load_data('question_1b', consumer_type)  # Load from question_1b folder
        else:  # question == '1c'
            data_loader._load_data('question_1c', consumer_type)  # Load from question_1c folder



        self.pv_capacity = data_loader.pv_max_power
        self.max_power_load = data_loader.max_power_load
        self.energy_prices = data_loader.energy_prices
        self.import_tariff = data_loader.import_tariff
        self.export_tariff = data_loader.export_tariff
        self.max_import = data_loader.max_import
        self.max_export = data_loader.max_export
        self.pv_hourly_ratio = data_loader.pv_hourly_ratio
        self.daily_load = data_loader.daily_load
        # Tariffs
        self.TOU_import_tariff_Radius = data_loader.TOU_radius
        self.TOU_import_tariff_N1 = data_loader.TOU_N1
        self.TOU_import_tariff_Bornholm = data_loader.TOU_bornholm
        #battery
        if question == '1a' or question == '1b':
            self.storage_capacity = None
            self.max_charging_power = None
            self.max_discharging_power = None
            self.charging_efficiency = None
            self.discharging_efficiency = None
            self.soc_init = None
            self.soc_final = None
        else:  # question == '1c'
            self.storage_capacity = data_loader.storage_capacity
            self.max_charging_power = data_loader.max_charging_power
            self.max_discharging_power = data_loader.max_discharging_power
            self.charging_efficiency = data_loader.charging_efficiency
            self.discharging_efficiency = data_loader.discharging_efficiency
            self.soc_init = data_loader.soc_init
            self.soc_final = data_loader.soc_final

        """
            elif question == '1a' or question == '1b':
            self.storage_capacity = None
            self.max_charging_power = None
            self.max_discharging_power = None
            self.charging_efficiency = None
            self.discharging_efficiency = None
            self.soc_init = None
            self.soc_final = None
"""
        

        # Load daily load requirement (different for 1a vs 1b)
        if question == '1a':
            self.daily_load = data_loader.daily_load  # Required daily energy
            self.hourly_preference = None  # No hourly preference profile
        else:  # question == '1b' or '1c'
            self.daily_load = None  # No daily requirement
            self.hourly_preference = data_loader.hourly_preference  # Load preference profile

        if tariff_scenario == 'import_tariff':
            self.active_tariff = [self.import_tariff] * 24  # Flat rate: [0.5, 0.5, 0.5, ...]
        elif tariff_scenario == 'TOU_import_tariff_Radius':
            self.active_tariff = self.TOU_import_tariff_Radius
        elif tariff_scenario == 'TOU_import_tariff_N1':
            self.active_tariff = self.TOU_import_tariff_N1
        elif tariff_scenario == 'TOU_import_tariff_Bornholm':
            self.active_tariff = self.TOU_import_tariff_Bornholm
        else:
            self.active_tariff = self.TOU_import_tariff_Radius  # Default


        
        self._build_model() # build gurobi model
    
    def _build_variables(self):
        

        # Only create decision variables: P_imp, P_exp, P_PV_prod
        self.P_imp = {t: self.model.addVar(lb=0, name=f'P_imp_{t}') for t in range(self.T)}
        self.P_exp = {t: self.model.addVar(lb=0, name=f'P_exp_{t}') for t in range(self.T)}
        self.P_PV_prod = {t: self.model.addVar(lb=0, name=f'P_PV_prod_{t}') for t in range(self.T)}
        
        # For question 1b, create D_hour and discomfort auxiliary variables
        if self.question == '1b':
            self.D_hour = {t: self.model.addVar(lb=0, ub=self.max_power_load, name=f'D_hour_{t}') for t in range(self.T)}
            # Add auxiliary variables for absolute value in discomfort calculation
            self.discomfort_pos = {t: self.model.addVar(lb=0, name=f'discomfort_pos_{t}') for t in range(self.T)}
            self.discomfort_neg = {t: self.model.addVar(lb=0, name=f'discomfort_neg_{t}') for t in range(self.T)}
        
        elif self.question == '1c':
            self.D_hour = {t: self.model.addVar(lb=0, ub=self.max_power_load, name=f'D_hour_{t}') for t in range(self.T)}
            # Add auxiliary variables for absolute value in discomfort calculation
            self.discomfort_pos = {t: self.model.addVar(lb=0, name=f'discomfort_pos_{t}') for t in range(self.T)}
            self.discomfort_neg = {t: self.model.addVar(lb=0, name=f'discomfort_neg_{t}') for t in range(self.T)}
            # Battery variables
            self.P_charge = {t: self.model.addVar(lb=0, ub=self.max_charging_power, name=f'P_charge_{t}') for t in range(self.T)}
            self.P_discharge = {t: self.model.addVar(lb=0, ub=self.max_discharging_power, name=f'P_discharge_{t}') for t in range(self.T)}
            self.SOC = {t: self.model.addVar(lb=0, ub=self.storage_capacity, name=f'SOC_{t}') for t in range(self.T)}

        # Store all variables in a single dictionary for compatibility
        self.variables = {}
        self.variables.update({f'P_imp_{t}': self.P_imp[t] for t in range(self.T)})
        self.variables.update({f'P_exp_{t}': self.P_exp[t] for t in range(self.T)})
        self.variables.update({f'P_PV_prod_{t}': self.P_PV_prod[t] for t in range(self.T)})

        if self.question == '1b':
            self.variables.update({f'D_hour_{t}': self.D_hour[t] for t in range(self.T)})

        elif self.question == '1c':
            self.variables.update({f'D_hour_{t}': self.D_hour[t] for t in range(self.T)})
            self.variables.update({f'P_charge_{t}': self.P_charge[t] for t in range(self.T)})
            self.variables.update({f'P_discharge_{t}': self.P_discharge[t] for t in range(self.T)})
            self.variables.update({f'SOC_{t}': self.SOC[t] for t in range(self.T)})
            

    def _build_constraints(self):
        if self.question == '1a':
        # Original Question 1a constraints
            self._build_constraints_1a()
        elif self.question == '1b':
        # New Question 1b constraints
            self._build_constraints_1b()
        else:  # question == '1c'
            self._build_constraints_1c()  # Start with 1b constraints


    def _build_constraints_1a(self):
        """Original constraints for Question 1a"""
        # Energy balance constraints: P_PV_prod_t + P_imp_t - P_exp_t = D_hour_t (implicit)
        self.energy_balance_constraints = {}
        for t in range(self.T):
            self.energy_balance_constraints[f'upper_{t}'] = self.model.addLConstr(
                self.P_PV_prod[t] + self.P_imp[t] - self.P_exp[t] <= self.max_power_load,
                name=f'energy_balance_upper_{t}'
            )
            self.energy_balance_constraints[f'lower_{t}'] = self.model.addLConstr(
                self.P_PV_prod[t] + self.P_imp[t] - self.P_exp[t] >= 0,
                name=f'energy_balance_lower_{t}'
            )

        # Daily demand constraint
        self.daily_demand_constraint = self.model.addLConstr(
            gp.quicksum(self.P_PV_prod[t] + self.P_imp[t] - self.P_exp[t] for t in range(self.T)) >= self.daily_load * self.max_power_load,
            name='daily_load'
        )
        
        # PV, import, export constraints (same as before)
        self._build_common_constraints()

    def _build_constraints_1b(self):
        """New constraints for Question 1b"""
        # 1. Energy balance: D_hour_t = P_PV_prod_t + P_imp_t - P_exp_t
        self.energy_balance_constraints = {}
        for t in range(self.T):
            self.energy_balance_constraints[t] = self.model.addLConstr(
                self.D_hour[t] == self.P_PV_prod[t] + self.P_imp[t] - self.P_exp[t],
                name=f'energy_balance_{t}'
            )

        # 2. Discomfort calculation: |D_hour_t - preferred_t|
        # We model this using: discomfort_pos_t - discomfort_neg_t = D_hour_t - preferred_t
        self.discomfort_constraints = {}
        for t in range(self.T):
            preferred_demand_t = self.hourly_preference[t] * self.max_power_load  # Scale preference
            
            self.discomfort_constraints[t] = self.model.addLConstr(
                self.discomfort_pos[t] - self.discomfort_neg[t] == self.D_hour[t] - preferred_demand_t,
                name=f'discomfort_{t}'
            )

        # 3. Common constraints (PV, import, export limits)
        self._build_common_constraints()

    def _build_constraints_1c(self):
        """New constraints for Question 1c"""
        # 1. Energy balance: D_hour_t = P_PV_prod_t + P_imp_t - P_exp_t + P_discharge_t - P_charge_t
        self.energy_balance_constraints = {}
        for t in range(self.T):
            self.energy_balance_constraints[t] = self.model.addLConstr(
                self.D_hour[t] == self.P_PV_prod[t] + self.P_imp[t] - self.P_exp[t] + self.P_discharge[t] - self.P_charge[t],
                name=f'energy_balance_{t}'
            )

        # 2. Discomfort calculation: |D_hour_t - preferred_t|
        # We model this using: discomfort_pos_t - discomfort_neg_t = D_hour_t - preferred_t
        self.discomfort_constraints = {}
        for t in range(self.T):
            preferred_demand_t = self.hourly_preference[t] * self.max_power_load  # Scale preference
            
            self.discomfort_constraints[t] = self.model.addLConstr(
                self.discomfort_pos[t] - self.discomfort_neg[t] == self.D_hour[t] - preferred_demand_t,
                name=f'discomfort_{t}'
            )
        # 3. Battery SOC dynamics and constraints
        self.soc_constraints = {}
        
        self.soc_constraints['init'] = self.model.addLConstr(
                self.SOC[0] == self.soc_init * self.storage_capacity + self.P_charge[0] * self.charging_efficiency - self.P_discharge[0] / self.discharging_efficiency,
                name='soc_initial'
            )
        
        self.soc_constraints['final'] = self.model.addLConstr(
                self.SOC[23] >= self.soc_final * self.storage_capacity,
                name=f'soc_final'
            )

        # SOC dynamics for hours 1-23
        for t in range(1, self.T):
            self.soc_constraints[t] = self.model.addLConstr(
                self.SOC[t] == self.SOC[t-1] + self.P_charge[t] * self.charging_efficiency - self.P_discharge[t] / self.discharging_efficiency,
                name=f'soc_dynamics_{t}'
            )
                # 3. Common constraints (PV, import, export limits)
        self._build_common_constraints()

    def _build_common_constraints(self):
            """Constraints common to all questions"""
            #PV production constraints: P_PV_prod_t <= pv_capacity * pv_ratio_t, remember that lb =0 in var def of P_PV_prod
            self.pv_production_constraints = {}
            for t in range(self.T):
                self.pv_production_constraints[t] = self.model.addLConstr(
                    self.P_PV_prod[t] <= self.pv_capacity * self.pv_hourly_ratio[t],
                    name=f'pv_production_upper{t}'
                )

                
            
            # 4. Import capacity constraints: P_imp_t <= max_import
            self.import_limit_constraints = {}
            for t in range(self.T):
                self.import_limit_constraints[t] = self.model.addLConstr(
                    self.P_imp[t] <= self.max_import,
                    name=f'import_limit_{t}'
                )
            
            # 5. Export capacity constraints: P_exp_t <= max_export
            self.export_limit_constraints = {}
            for t in range(self.T):
                self.export_limit_constraints[t] = self.model.addLConstr(
                    self.P_exp[t] <= self.max_export,
                    name=f'export_limit_{t}'
                )
            
            # Store all constraints in a list for compatibility
            self.constraints = []
            self.constraints.extend(list(self.energy_balance_constraints.values()))
            if hasattr(self, 'daily_demand_constraint'):
                self.constraints.append(self.daily_demand_constraint)
            if hasattr(self, 'discomfort_constraints'):
                self.constraints.extend(list(self.discomfort_constraints.values()))
            if hasattr(self, 'soc_constraints'):  # ADD THIS
                self.constraints.extend(list(self.soc_constraints.values()))
            self.constraints.extend(list(self.pv_production_constraints.values()))
            self.constraints.extend(list(self.import_limit_constraints.values()))
            self.constraints.extend(list(self.export_limit_constraints.values()))


    def _build_objective_function(self):
        if self.question == '1a':
            # Objective: Minimize total cost = sum(P_imp_t * (energy_price + import_tariff))
            objective = gp.quicksum(
                self.P_imp[t] * (self.energy_prices[t] + self.active_tariff[t])
                for t in range(self.T)
            )
        elif self.question == '1b' or self.question == '1c':
        # New objective: minimize import costs + discomfort penalty
            import_cost = gp.quicksum(
                self.P_imp[t] * (self.energy_prices[t] + self.active_tariff[t])
                for t in range(self.T)
            )
            
            # Discomfort penalty (quadratic approximation using linear terms)
            discomfort_penalty = gp.quicksum(
                self.alpha_discomfort * (self.discomfort_pos[t] + self.discomfort_neg[t])
                for t in range(self.T)
            )
        
            objective = import_cost + discomfort_penalty
        


        self.model.setObjective(objective, GRB.MINIMIZE)


    def _build_model(self):
        self.model = gp.Model(name='Minimize energy procurement')
        self._build_variables()
        self._build_constraints()
        self._build_objective_function()
        self.model.update()

    def _save_results(self):
        """Save results and calculate D_hour from energy balance"""
        self.results.objective_value = self.model.ObjVal
        
        # Save decision variables
        self.results.P_imp = {t: self.P_imp[t].x for t in range(self.T)}
        self.results.P_exp = {t: self.P_exp[t].x for t in range(self.T)}
        self.results.P_PV_prod = {t: self.P_PV_prod[t].x for t in range(self.T)}
        
        if self.question == '1a':
            # Calculate D_hour from energy balance
            self.results.D_hour = {t: self.results.P_PV_prod[t] + self.results.P_imp[t] - self.results.P_exp[t] 
                                for t in range(self.T)}
        elif self.question == '1b':
            # D_hour is a decision variable
            self.results.D_hour = {t: self.D_hour[t].x for t in range(self.T)}
            
            # Calculate discomfort metrics
            self.results.discomfort_pos = {t: self.discomfort_pos[t].x for t in range(self.T)}
            self.results.discomfort_neg = {t: self.discomfort_neg[t].x for t in range(self.T)}
            
            # Calculate total discomfort
            total_discomfort = sum(self.results.discomfort_pos[t] + self.results.discomfort_neg[t] 
                                for t in range(self.T))
            self.results.total_discomfort = total_discomfort

        else:  # question == '1c'
            # D_hour is a decision variable
            self.results.D_hour = {t: self.D_hour[t].x for t in range(self.T)}
            
            # Calculate discomfort metrics
            self.results.discomfort_pos = {t: self.discomfort_pos[t].x for t in range(self.T)}
            self.results.discomfort_neg = {t: self.discomfort_neg[t].x for t in range(self.T)}
            
            # Calculate total discomfort
            total_discomfort = sum(self.results.discomfort_pos[t] + self.results.discomfort_neg[t] 
                                for t in range(self.T))
            self.results.total_discomfort = total_discomfort

            # Save battery results
            self.results.P_charge = {t: self.P_charge[t].x for t in range(self.T)}
            self.results.P_discharge = {t: self.P_discharge[t].x for t in range(self.T)}
            self.results.SOC = {t: self.SOC[t].x for t in range(self.T)}
        # Save dual values
        self.results.duals = [const.Pi for const in self.constraints]


    def run(self):
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            self._save_results()
        else:
            print(f"optimization of {self.model.ModelName} was not successful")

    def display_results(self):
        print("-------------------   RESULTS  -------------------")
        print(f"Optimal objective value: {self.results.objective_value:.4f}")
        
        print("\nHourly Import Power (kW):")
        for t, value in self.results.P_imp.items():
            print(f"  Hour {t:2d}: {value:.4f}")
        
        print("\nHourly Export Power (kW):")
        for t, value in self.results.P_exp.items():
            print(f"  Hour {t:2d}: {value:.4f}")
        
        print("\nHourly PV Production (kW):")
        for t, value in self.results.P_PV_prod.items():
            print(f"  Hour {t:2d}: {value:.4f}")
        
        print("\nHourly Demand (Calculated from energy balance) (kW):")
        for t, value in self.results.D_hour.items():
            print(f"  Hour {t:2d}: {value:.4f}")

        if self.question == '1b':
            print("\nHourly Discomfort Positive Deviation (kW):")
            for t, value in self.results.discomfort_pos.items():
                print(f"  Hour {t:2d}: {value:.4f}")
            
            print("\nHourly Discomfort Negative Deviation (kW):")
            for t, value in self.results.discomfort_neg.items():
                print(f"  Hour {t:2d}: {value:.4f}")
            
            print(f"\nTotal Discomfort: {self.results.total_discomfort:.4f} kW")
            #print preferred demand
            print("\nPreferred Hourly Demand (kW):")
            for t in range(self.T):
                preferred_demand_t = self.hourly_preference[t] * self.max_power_load
                print(f"  Hour {t:2d}: {preferred_demand_t:.4f}")

        if self.question == '1c':
            print("\nHourly Discomfort Positive Deviation (kW):")
            for t, value in self.results.discomfort_pos.items():
                print(f"  Hour {t:2d}: {value:.4f}")
            
            print("\nHourly Discomfort Negative Deviation (kW):")
            for t, value in self.results.discomfort_neg.items():
                print(f"  Hour {t:2d}: {value:.4f}")
            
            print(f"\nTotal Discomfort: {self.results.total_discomfort:.4f} kW")
            #print preferred demand
            print("\nPreferred Hourly Demand (kW):")
            for t in range(self.T):
                preferred_demand_t = self.hourly_preference[t] * self.max_power_load
                print(f"  Hour {t:2d}: {preferred_demand_t:.4f}")

            print("\nHourly Battery Charging Power (kW):")
            for t, value in self.results.P_charge.items():
                print(f"  Hour {t:2d}: {value:.4f}")

            print("\nHourly Battery Discharging Power (kW):")
            for t, value in self.results.P_discharge.items():
                print(f"  Hour {t:2d}: {value:.4f}")

            print("\nHourly State of Charge (kWh):")
            for t, value in self.results.SOC.items():
                print(f"  Hour {t:2d}: {value:.4f}")

        
        print(f"\nTotal daily demand: {sum(self.results.D_hour.values()):.4f} kWh")
        print("--------------------------------------------------")

"""
#Lesias kode:

    def _build_objective_function(self):
        objective = gp.quicksum(self.data.objective_coeff[v] * self.variables[v] for v in self.data.VARIABLES)
        self.model.setObjective(objective, GRB.MINIMIZE)

    def _build_model(self):
        self.model = gp.Model(name='Economic dispatch')
        self._build_variables()
        self._build_objective_function()
        self._build_constraints()
        self.model.update()
    
    def _save_results(self):
        self.results.objective_value = self.model.ObjVal
        self.results.variables = {v: self.variables[v].x for v in self.data.VARIABLES}
        self.results.duals = [self.constraints[i].Pi for i in range(len(self.constraints))]

    def run(self):
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            self._save_results()
        else:
            print(f"optimization of {model.ModelName} was not successful")
    
    def display_results(self):
        print()
        print("-------------------   RESULTS  -------------------")
        print("Optimal objective value:")
        print(self.results.objective_value)
        print("Optimal variable values:")
        print(self.results.variables)
        print("Optimal dual values:")
        print(self.results.duals)
"""