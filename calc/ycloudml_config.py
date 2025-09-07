import os

"""
Configuration file for YCloudML Impact Analysis
"""

# Set this to your project root folder
PROJECT_ROOT = r"C:\Users\Valentin\Desktop\data_for_impact_analysis"
PROJECT_SUBFOLDER = r"data\inTouch"  # Change if your subfolder structure is different
PROJECT_PATH = os.path.join(PROJECT_ROOT, PROJECT_SUBFOLDER)

# Yandex Cloud Configuration
YANDEX_CONFIG = {
    "folder_id": "b1g4anao7mgsjdv18cgi",  # Your Yandex Cloud folder ID
    "auth_token": "t1.9euelZqbl5mTjMnGnM_HjMbHks-Pmu3rnpWax5PLmJmOk5jHkpLOl4nLz43l9Pc1WHE7-e9XMRf43fT3dQZvO_nvVzEX-M3n9euelZqQkJ7JnpzJyMzPjsiUxo7LkO_8xeuelZqQkJ7JnpzJyMzPjsiUxo7LkA.JdLtW9VcMcDnz4GhgUPTGYHaJsT2w0qrC4WX-xi2W3LFYCSwCR7lHMFtv0Rr05ACfWryt2Dc_ND1rssFyIWwCA",
    "model_name": "llama-lite",            
    "temperature": 0.3                    
}

# File Paths
MODEL_DIR = os.path.join(PROJECT_PATH, "models", "selt.txt")
CHANGE_REQUESTS_PATH = os.path.join(PROJECT_PATH, "change_requests.txt")
PROJECT_DESCRIPTION_PATH = os.path.join(PROJECT_PATH, "project_description.txt")
OUTPUTS_DIR = os.path.join(PROJECT_PATH, "outputs")

with open(PROJECT_DESCRIPTION_PATH, 'r') as f:
    pr_desc = f.read()

# Project Description
PROJECT_DESCRIPTION = pr_desc

# Analysis Parameters
ANALYSIS_CONFIG = {
    "num_cycles": 1,
    "pause_between_cycles": 0.1,
    "pause_between_requests": 0.1
}

# Extract model name from model directory
MODEL_NAME = os.path.basename(MODEL_DIR)
# Extract project name from subfolder
PROJECT_NAME = os.path.basename(PROJECT_PATH)
# Output Configuration
OUTPUT_CONFIG = {
    "csv_filename": f"{MODEL_NAME}_{YANDEX_CONFIG['model_name']}_t{YANDEX_CONFIG['temperature']}_cyc{ANALYSIS_CONFIG['num_cycles']}.csv",
    "outputs_dir": OUTPUTS_DIR
}

# FILE_PATHS for backward compatibility
FILE_PATHS = {
    "model_path": MODEL_DIR,
    "change_requests_path": CHANGE_REQUESTS_PATH
}

# Parameter grid for grid search
PARAM_GRID = {
    "model_name": ["llama"],
    "context_type": [
        "components",
        "dsm",
        "dsmExtended"
    ]
}

# Map context_type to actual file paths
CONTEXT_PATHS = {
    "components": os.path.join(PROJECT_PATH, "models", "components.txt"),
    "dsm": os.path.join(PROJECT_PATH, "models", "dsm.txt"),
    "dsmExtended": os.path.join(PROJECT_PATH, "models", "dsm-extended.txt"),
    "selt": os.path.join(PROJECT_PATH, "models", "selt.txt"),
}