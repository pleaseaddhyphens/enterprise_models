"""
Configuration file for YCloudML Impact Analysis
"""

# Yandex Cloud Configuration
YANDEX_CONFIG = {
    "folder_id": "b1g4anao7mgsjdv18cgi",  # Your Yandex Cloud folder ID
    "auth_token": "t1.9euelZrOmseezZKZipiczJiYj5Kenu3rnpWax5PLmJmOk5jHkpLOl4nLz43l8_dQazE8-e89Lygi_d3z9xAaLzz57z0vKCL9zef1656Vmo6Sx8qVk5fIlImSz8-VjYqb7_zF656Vmo6Sx8qVk5fIlImSz8-VjYqb.kaMyRzF-87toIDhwoCqKbLibjQrpVlWpW1K5Y2V4VawgJ1I97riutFb_EMjIGau7tz-LxgBslEMVK72RkJBNCA",   # Your IAM token
    "model_name": "llama-lite",            # Model to use
    "temperature": 0.3                     # Model temperature
}

# File Paths
FILE_PATHS = {
    "model_path": r"C:\Users\Valentin\Desktop\data_for_impact_analysis\data\inTouch\export.txt",
    "change_requests_path": r"C:\Users\Valentin\Desktop\data_for_impact_analysis\data\inTouch\change_requests.txt"
}

# Project Description
PROJECT_DESCRIPTION = "Permafrost regions contain numerous natural gas deposits. Pipelines transport gas from extraction sites to end users. The pipeline supports (columns/piles) are subject to displacement due to permafrost thawing processes and frost heave which can lead to pipeline deformations and damage.Currently pipeline integrity inspections are primarily conducted manually requiring significant time and labor resources. Moreover this approach does not always ensure timely defect detection. The team developing the technology to monitor pipeline deformations using fiber Bragg grating (FBG) sensors."

# Analysis Parameters
ANALYSIS_CONFIG = {
    "num_cycles": 1,
    "pause_between_cycles": 0.1,
    "pause_between_requests": 0.1
}

# Output Configuration
OUTPUT_CONFIG = {
    "csv_filename": "llama8b-yandex-1cycle.csv",
    "outputs_dir": r"C:\Users\Valentin\Desktop\data_for_impact_analysis\data\inTouch\outputs"
} 