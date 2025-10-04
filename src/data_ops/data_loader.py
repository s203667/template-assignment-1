# -----------------------------
# Load Data
# -----------------------------
import json
import pandas as pd
from pathlib import Path


class DataLoader:

    def __init__(self, input_path: str):
        # Convert to Path object and resolve absolute path 
        self.input_path = Path(input_path).resolve()
        


    def _load_data_file(self, question_name: str, file_name: str):
        """
        Helper function to load a specific JSON file, and store it as a class attribute.
        
        Parameters:
        - question_name: Subdirectory under input_path
        - file_name: Name of the file (CSV or JSON)
        
        Returns:
        - loaded data (dict for JSON)
        """
        file_path = self.input_path / question_name / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Load JSON
        if file_path.suffix == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
                return data
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

class InputData:

    def __init__(
        self, 
        VARIABLES: list,
        objective_coeff: list[str, int],    # Coefficients in objective function
        constraints_coeff: list[str, int],  # Linear coefficients of constraints
        constraints_rhs: list[str, int],    # Right hand side coefficients of constraints
        constraints_sense: list[str, int],  # Direction of constraints
    ):
        self.VARIABLES = VARIABLES
        self.objective_coeff = objective_coeff
        self.constraints_coeff = constraints_coeff
        self.constraints_rhs = constraints_rhs
        self.constraints_sense = constraints_sense



