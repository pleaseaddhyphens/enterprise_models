import pandas as pd
import numpy as np
import os
import json
from utils import calculate_alignment_scores, calculate_novelty_scores, calculate_f_scores

def calculate_scores_variance(csv_path, change_requests_path, human_df, score_type, relevancy):
    """
    Calculate scores for all cycles in the CSV file and compute cycle-to-cycle variance and more.
    
    Args:
        csv_path: Path to the CSV file containing all cycles data
        change_requests_path: Path to change requests file
        human_df: Engineer dataframe (fixed human responses)
        
    Returns:
        dict: A dictionary containing:
            - 'all_cycle_results': List of cycle results with scores and averages
            - 'overall_average': Average across all cycles
            - 'cycle_to_cycle_std': Standard deviation of cycle averages
            - 'overall_scores_std':
    """
    print(f"Score type: {score_type}")
    # Read the entire CSV file
    full_df = pd.read_csv(csv_path)
    
    # Get unique cycles
    unique_cycles = full_df['Cycle'].unique()
    
    all_cycle_results = []
    cycle_averages = []  # To collect cycle averages for cycle-to-cycle std calculation
    scores = []

    for cycle in unique_cycles:
        # Filter dataframe for the current cycle
        cycle_df = full_df[full_df['Cycle'] == cycle]
        
        # Calculate alignment scores for this cycle
        if score_type == 'alignment':
            scores, avg_score = calculate_alignment_scores(change_requests_path, cycle_df, human_df)
            
            # Store results for this cycle
            all_cycle_results.append({
                'cycle': int(cycle),
                'scores': scores,
                'average': avg_score,
                'std': round(np.std(scores),3)
            })
            # Collect cycle averages for cycle-to-cycle calculations
            cycle_averages.append(avg_score)
            scores.extend(scores)
        if score_type == 'diversity':
            cycle_df['Relevant'] = 'Yes'
            scores, avg_score = calculate_novelty_scores(cycle_df, human_df)
            avg_score = avg_score*relevancy
            
            all_cycle_results.append({
                'cycle': int(cycle),
                'scores': scores,
                'average': avg_score,
                'std': round(np.std(scores),3)
            })
            # Collect cycle averages for cycle-to-cycle calculations
            cycle_averages.append(avg_score)
            scores.extend(scores)
        if score_type == 'f':
            scores, avg_score, _ = calculate_f_scores(change_requests_path,cycle_df, human_df )
            all_cycle_results.append({
                'cycle': int(cycle),
                'scores': scores,
                'average': avg_score,
                'std': round(np.std(scores),3),
                'var': round(np.var(scores),3)
            })
            # Collect cycle averages for cycle-to-cycle calculations
            cycle_averages.append(avg_score)
            scores.extend(scores)

        else:
            pass


    
    # Calculate overall metrics
    overall_avg = np.mean(cycle_averages) if cycle_averages else 0
    cycle_to_cycle_std = np.std(cycle_averages) if cycle_averages else 0
    cycle_to_cycle_var = np.var(cycle_averages) if cycle_averages else 0

    scores_std = np.std(scores)if scores else 0
    return {
        'all_cycle_results': all_cycle_results,
        'overall_average': round(overall_avg,3),
        'cycle_to_cycle_std': round(cycle_to_cycle_std,3),
        'cycle_to_cycle_var': round(cycle_to_cycle_var,3),
        'overall_scores_std': round(scores_std,2)

    }

PROJECT_PATH = r"C:\Users\Valentin\Desktop\data_for_impact_analysis\data\aq"

CHANGE_REQUESTS_PATH = os.path.join(PROJECT_PATH, "change_requests.txt")
OUTPUTS_DIR = os.path.join(PROJECT_PATH, "outputs")
TEAM_DATA = os.path.join(PROJECT_PATH, "team_data")

# Example usage:
human_df = pd.read_csv(f'{TEAM_DATA}\\orange_valya.txt')  # Load your human responses
llm_csv = f"{OUTPUTS_DIR}\\selt_llama_t0.3_cyc10.csv"
results = calculate_scores_variance(llm_csv, CHANGE_REQUESTS_PATH, human_df, "f", relevancy=1)
print(json.dumps(results,  indent=4))
