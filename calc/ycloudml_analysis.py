#!/usr/bin/env python3
"""
YCloudML-based Impact Analysis Script

This script uses YCloudML as the LLM endpoint for conducting impact analysis.
It processes change requests and saves both response text files and structured data.
"""

import os
import re
import pandas as pd
import time
from yandex_cloud_ml_sdk import YCloudML
from utils import (
    load_model_data, 
    load_change_requests, 
    extract_suggestions, 
    create_prompt, 
    parse_response
)

class YCloudMLAnalyzer:
    """Handles impact analysis using YCloudML as the LLM endpoint."""
    
    def __init__(self, folder_id, auth_token, model_name='llama-lite', temperature=0.3):
        """
        Initialize the YCloudML analyzer.
        
        Args:
            folder_id: Yandex Cloud folder ID
            auth_token: IAM token for authentication
            model_name: Model to use (default: 'llama-lite')
            temperature: Model temperature setting
        """
        self.sdk = YCloudML(folder_id=folder_id, auth=auth_token)
        self.model = self.sdk.models.completions(model_name)
        self.model = self.model.configure(temperature=temperature)
        
    def get_response(self, prompt):
        """
        Get response from YCloudML model.
        
        Args:
            prompt: The prompt to send to the model
            
        Returns:
            str: The model's response text
        """
        try:
            result = self.model.run({"role": "user", "text": prompt})
            
            # Extract the response text from the result
            response_text = ""
            for alternative in result:
                if hasattr(alternative, 'text'):
                    response_text = alternative.text
                    break
                elif hasattr(alternative, 'content'):
                    response_text = alternative.content
                    break
            
            return response_text
        except Exception as e:
            print(f"Error getting response from YCloudML: {e}")
            return ""



def process_single_change_request_ycloudml(
    analyzer, 
    model_data, 
    change_request, 
    change_request_id, 
    cycle_number, 
    project_description
):
    """
    Process a single change request using YCloudML.
    
    Args:
        analyzer: YCloudMLAnalyzer instance
        model_data: The model data as string
        change_request: The change request text
        change_request_id: Numeric ID for the change request
        cycle_number: Current cycle number
        project_description: Project description
        
    Returns:
        pd.DataFrame with impact analysis results
    """
    prompt = create_prompt(model_data, change_request, project_description)
    
    print(f"Processing change request {change_request_id} (cycle {cycle_number})...")
    
    # Get response from YCloudML
    response_text = analyzer.get_response(prompt)
    
    # Parse response into DataFrame
    df = parse_response(response_text, change_request_id, cycle_number)
    
    return df

def process_all_change_requests_ycloudml(
    analyzer,
    model_path, 
    change_requests_path, 
    cycle_number, 
    project_description, 
    pause=2
):
    """
    Process all change requests using YCloudML.
    
    Args:
        analyzer: YCloudMLAnalyzer instance
        model_path: Path to model data file
        change_requests_path: Path to change requests file
        cycle_number: Current cycle number
        project_description: Project description
        pause: Seconds to pause between requests
        
    Returns:
        Combined DataFrame with all impacts
    """
    model_data = load_model_data(model_path)
    change_requests = load_change_requests(change_requests_path)
    
    combined_df = pd.DataFrame()
    
    for idx, change_request in enumerate(change_requests, start=1):
        df = process_single_change_request_ycloudml(
            analyzer=analyzer,
            model_data=model_data,
            change_request=change_request,
            change_request_id=idx,
            cycle_number=cycle_number,
            project_description=project_description
        )
        
        if not df.empty:
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        
        if idx < len(change_requests):
            print(f"Waiting {pause} seconds before next request...")
            time.sleep(pause)
    
    return combined_df

def run_full_analysis_ycloudml(
    model_path, 
    change_requests_path, 
    project_description,
    folder_id,
    auth_token,
    model_name='llama-lite',
    temperature=0.3,
    num_cycles=1, 
    pause_between_cycles=3, 
    pause_between_requests=2
):
    """
    Run full analysis with multiple cycles using YCloudML.
    
    Args:
        model_path: Path to model data file
        change_requests_path: Path to change requests file
        project_description: Project description
        folder_id: Yandex Cloud folder ID
        auth_token: IAM token for authentication
        model_name: Model to use
        temperature: Model temperature setting
        num_cycles: Number of cycles to run
        pause_between_cycles: Seconds between cycles
        pause_between_requests: Seconds between requests
        
    Returns:
        Combined DataFrame with all impacts from all cycles
    """
    # Initialize YCloudML analyzer
    analyzer = YCloudMLAnalyzer(
        folder_id=folder_id,
        auth_token=auth_token,
        model_name=model_name,
        temperature=temperature
    )
    
    all_runs_df = pd.DataFrame()

    for cycle in range(1, num_cycles + 1):
        print(f"\n=== Running cycle {cycle}/{num_cycles} ===")
        
        df = process_all_change_requests_ycloudml(
            analyzer=analyzer,
            model_path=model_path,
            change_requests_path=change_requests_path,
            cycle_number=cycle,
            project_description=project_description,
            pause=pause_between_requests
        )
        
        all_runs_df = pd.concat([all_runs_df, df], ignore_index=True)
        
        if cycle < num_cycles:
            print(f"Waiting {pause_between_cycles} seconds before next cycle...")
            time.sleep(pause_between_cycles)
    
    return all_runs_df

def main():
    """Main function to run the YCloudML analysis."""
    
    # Import configuration
    from ycloudml_config import YANDEX_CONFIG, FILE_PATHS, PROJECT_DESCRIPTION, ANALYSIS_CONFIG, OUTPUT_CONFIG
    
    # Configuration from config file
    FOLDER_ID = YANDEX_CONFIG["folder_id"]
    AUTH_TOKEN = YANDEX_CONFIG["auth_token"]
    MODEL_NAME = YANDEX_CONFIG["model_name"]
    TEMPERATURE = YANDEX_CONFIG["temperature"]
    
    # File paths
    MODEL_PATH = FILE_PATHS["model_path"]
    CHANGE_REQUESTS_PATH = FILE_PATHS["change_requests_path"]
    
    # Project description
    PROJECT_DESCRIPTION = PROJECT_DESCRIPTION
    
    # Analysis parameters
    NUM_CYCLES = ANALYSIS_CONFIG["num_cycles"]
    PAUSE_BETWEEN_CYCLES = ANALYSIS_CONFIG["pause_between_cycles"]
    PAUSE_BETWEEN_REQUESTS = ANALYSIS_CONFIG["pause_between_requests"]
    
    try:
        print("Starting YCloudML Impact Analysis...")
        print(f"Using model: {MODEL_NAME}")
        
        # Run the analysis
        df = run_full_analysis_ycloudml(
            model_path=MODEL_PATH,
            change_requests_path=CHANGE_REQUESTS_PATH,
            project_description=PROJECT_DESCRIPTION,
            folder_id=FOLDER_ID,
            auth_token=AUTH_TOKEN,
            model_name=MODEL_NAME,
            temperature=TEMPERATURE,
            num_cycles=NUM_CYCLES,
            pause_between_cycles=PAUSE_BETWEEN_CYCLES,
            pause_between_requests=PAUSE_BETWEEN_REQUESTS
        )
        
        # Save results
        if not df.empty:
            output_csv = os.path.join(OUTPUT_CONFIG["outputs_dir"], OUTPUT_CONFIG["csv_filename"])
            os.makedirs(OUTPUT_CONFIG["outputs_dir"], exist_ok=True)
            df.to_csv(output_csv, index=False)
            print(f"\nAnalysis completed successfully!")
            print(f"Results saved to: {output_csv}")
            print(f"Total impacts identified: {len(df)}")
        else:
            print("No results generated. Check the response files for errors.")
            
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main() 