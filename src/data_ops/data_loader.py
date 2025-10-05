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
        # Initialize attributes to None (good practice)
        self.app_data1 = None
        self.max_power_load = None
        self.bus_params = None
        self.consumer_params = None
        self.DER_production = None
        self.usage_preference = None
        


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


    def _load_data(self, question='question_1a'):
        """
        Load all required data files and store them as class attributes.
        """
        # Load each data file
        self.app_data1 = self._load_data_file(question, 'appliance_params.json')
        self.bus_params = self._load_data_file(question, 'bus_params.json')
        self.consumer_params = self._load_data_file(question, 'consumer_params.json')
        self.DER_production = self._load_data_file(question, 'DER_production.json')
        self.usage_preference = self._load_data_file(question, 'usage_preference.json')

        self.max_power_load = self.app_data1['load'][0]['max_load_kWh_per_hour'] 
        self.pv_max_power = self.app_data1['DER'][0]['max_power_kW']
        self.energy_prices = self.bus_params[0]['energy_price_DKK_per_kWh']
        self.import_tariff = self.bus_params[0]['import_tariff_DKK/kWh']
        self.export_tariff = self.bus_params[0]['export_tariff_DKK/kWh']
        self.max_import = self.bus_params[0]['max_import_kW']
        self.max_export = self.bus_params[0]['max_export_kW']
        self.pv_hourly_ratio = self.DER_production[0]['hourly_profile_ratio']
        self.daily_load = self.usage_preference[0]['load_preferences'][0]['min_total_energy_per_day_hour_equivalent']

        self.TOU_radius = self.bus_params[0]['import_tariff_time_of_use_radius']
        self.TOU_N1 = self.bus_params[0]['import_tariff_time_of_use_N1']
        self.TOU_bornholm = self.bus_params[0]['import_tariff_time_of_use_treforbornholm']
        
        # Load daily requirement or hourly preferences based on question
        try:
            # For Question 1a: daily energy requirement
            self.daily_load = self.usage_preference[0]['load_preferences'][0]['min_total_energy_per_day_hour_equivalent']
        except (KeyError, TypeError):
            self.daily_load = None
        
        try:
            # For Question 1b: hourly preference profile
            self.hourly_preference = self.usage_preference[0]['load_preferences'][0]['hourly_profile_ratio']
        except (KeyError, TypeError):
            self.hourly_preference = None
    


# Test the DataLoader
if __name__ == "__main__":
    # Test loading
    data_loader = DataLoader('../../data')  # Changed from '../data' to '../../data'
    data_loader._load_data()
    
    print(f"Max power load: {data_loader.max_power_load}")
    print(f"daily load: {data_loader.daily_load}")
    #print(f"PV max power: {data_loader.pv_max_power}")
    #print(f"Energy prices (first 5): {data_loader.energy_prices[:5]}")
#Call energy_prices from _load_data
#DataLoader._load_data()



#print(self.app_data1['DER'][0]['max_power_kW'])
#energy_prices = self.bus_params[0]['max_import_kW']
#print(energy_prices)


