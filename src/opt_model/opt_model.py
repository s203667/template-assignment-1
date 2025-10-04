from pathlib import Path
import numpy as np
import pandas as pd
import gurobipy as gp
import xarray as xr

from data_ops.data_loader import DataLoader
from data_ops.data_loader import InputData



class Expando(object):
    '''
        A small class which can have attributes set
    '''
    pass


class OptModel():

    def __init__(self, input_data: InputData): # initialize class
        self.data = input_data # define data attributes
        self.results = Expando() # define results attributes
        self._build_model() # build gurobi model
    
    def _build_variables(self):
        self.variables = {v: self.model.addVar(lb=0, name='Total production of CHP {0}'.format(v)) for v in self.data.VARIABLES}
    
    def _build_constraints(self):
        self.constraints = [
            (
                self.model.addLConstr(
                        gp.quicksum(self.data.constraints_coeff[v][i] * self.variables[v] for v in self.data.VARIABLES),
                        self.data.constraints_sense[i],
                        self.data.constraints_rhs[i]
                )
            ) for i in range(len(self.data.constraints_rhs))
        ]

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