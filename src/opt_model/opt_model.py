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

data_loader = DataLoader('../data')  # Changed from '../data' to '../../data'

class Expando(object):
    '''
        A small class which can have attributes set
    '''
    pass


class OptModel():

    def __init__(self): # initialize class
        #self.data = input_data # define data attributes
        self.results = Expando() # define results attributes
        # Load data once and store as instance attributes
        self.T = 24
        data_loader._load_data()
        self.pv_capacity = data_loader.pv_max_power
        self.max_power_load = data_loader.max_power_load
        self.energy_prices = data_loader.energy_prices
        self.import_tariff = data_loader.import_tariff
        self.export_tariff = data_loader.export_tariff
        self.max_import = data_loader.max_import
        self.max_export = data_loader.max_export
        self.pv_hourly_ratio = data_loader.pv_hourly_ratio
        self.daily_load = data_loader.daily_load
        
        self._build_model() # build gurobi model
    
    def _build_variables(self):
        

        # Only create decision variables: P_imp, P_exp, P_PV_prod
        self.P_imp = {t: self.model.addVar(lb=0, name=f'P_imp_{t}') for t in range(self.T)}
        self.P_exp = {t: self.model.addVar(lb=0, name=f'P_exp_{t}') for t in range(self.T)}
        self.P_PV_prod = {t: self.model.addVar(lb=0, name=f'P_PV_prod_{t}') for t in range(self.T)}
        
        # Store all variables in a single dictionary for compatibility
        self.variables = {}
        self.variables.update({f'P_imp_{t}': self.P_imp[t] for t in range(self.T)})
        self.variables.update({f'P_exp_{t}': self.P_exp[t] for t in range(self.T)})
        self.variables.update({f'P_PV_prod_{t}': self.P_PV_prod[t] for t in range(self.T)})

    def _build_constraints(self):

       # 1. Energy balance constraints: P_PV_prod_t + P_imp_t - P_exp_t = D_hour_t
        # where D_hour_t is limited to max_power_load (3 kW)
        self.energy_balance_constraints = {}
        for t in range(self.T):
            # D_hour_t is constrained between 0 and max_power_load
            # Energy balance: P_PV_prod_t + P_imp_t - P_exp_t <= max_power_load
            self.energy_balance_constraints[f'upper_{t}'] = self.model.addLConstr(
                self.P_PV_prod[t] + self.P_imp[t] - self.P_exp[t] <= self.max_power_load,
                name=f'energy_balance_upper_{t}'
            )
            # And: P_PV_prod_t + P_imp_t - P_exp_t >= 0 (non-negative demand)
            self.energy_balance_constraints[f'lower_{t}'] = self.model.addLConstr(
                self.P_PV_prod[t] + self.P_imp[t] - self.P_exp[t] >= 0,
                name=f'energy_balance_lower_{t}'
            )

        # 2. Daily demand constraint: sum(D_hour_t) >= D_daily
        # Since D_hour_t = P_PV_prod_t + P_imp_t - P_exp_t, this becomes:
        # sum(P_PV_prod_t + P_imp_t - P_exp_t) >= D_daily
        self.daily_demand_constraint = self.model.addLConstr(
            gp.quicksum(self.P_PV_prod[t] + self.P_imp[t] - self.P_exp[t] for t in range(self.T)) >= self.daily_load * self.max_power_load,
            name='daily_load'
        )
        # 3. PV production constraints: P_PV_prod_t = pv_capacity * pv_ratio_t
        self.pv_production_constraints = {}
        for t in range(self.T):
            self.pv_production_constraints[t] = self.model.addLConstr(
                self.P_PV_prod[t] == self.pv_capacity * self.pv_hourly_ratio[t],
                name=f'pv_production_{t}'
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
        self.constraints.append(self.daily_demand_constraint)
        self.constraints.extend(list(self.pv_production_constraints.values()))
        self.constraints.extend(list(self.import_limit_constraints.values()))
        self.constraints.extend(list(self.export_limit_constraints.values()))


    def _build_objective_function(self):
        # Objective: Minimize total cost = sum(P_imp_t * (energy_price + import_tariff))
        objective = gp.quicksum(
            self.P_imp[t] * (self.energy_prices[t] + self.import_tariff)
            for t in range(self.T)
        )
        self.model.setObjective(objective, GRB.MINIMIZE)
 

    def _build_model(self):
        self.model = gp.Model(name='Minimize energy procurement')
        self._build_variables()
        self._build_objective_function()
        self._build_constraints()
        self.model.update()

    def _save_results(self):
        """Save results and calculate D_hour from energy balance"""
        self.results.objective_value = self.model.ObjVal
        
        # Save decision variables
        self.results.P_imp = {t: self.P_imp[t].x for t in range(self.T)}
        self.results.P_exp = {t: self.P_exp[t].x for t in range(self.T)}
        self.results.P_PV_prod = {t: self.P_PV_prod[t].x for t in range(self.T)}
        
        # Calculate D_hour from energy balance (as auxiliary values)
        self.results.D_hour = {t: self.results.P_PV_prod[t] + self.results.P_imp[t] - self.results.P_exp[t] 
                              for t in range(self.T)}
        
        # Store in original format for compatibility
        self.results.variables = {}
        self.results.variables.update({f'P_imp_{t}': self.P_imp[t].x for t in range(self.T)})
        self.results.variables.update({f'P_exp_{t}': self.P_exp[t].x for t in range(self.T)})
        self.results.variables.update({f'P_PV_prod_{t}': self.P_PV_prod[t].x for t in range(self.T)})

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