from pathlib import Path
import numpy as np
import pandas as pd
import gurobipy as gp
import xarray as xr

from data_ops import data_loader


class OptModel:
    """
    Placeholder for optimization models using Gurobipy.

    Attributes (examples):
        N (int): Number of time steps/consumers/etc.
        question/scenario name (str): Configuration/question identifier.
        ...
    """