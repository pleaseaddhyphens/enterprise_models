import pandas as pd
import numpy as np
from utils import calculate_alignment_scores

def calculate_alignment_scores_for_all_cycles(csv_path, change_requests_path, human_df):
    """
    Calculate alignment scores for all cycles in the CSV file and compute cycle-to-cycle variance.
    
    Args:
        csv_path: Path to the CSV file containing all cycles data
        change_requests_path: Path to change requests file
        human_df: Engineer dataframe (fixed human responses)
        
    Returns:
        dict: A dictionary containing:
            - 'all_cycle_results': List of cycle results with scores and averages
            - 'overall_average': Average across all cycles
            - 'cycle_to_cycle_std': Standard deviation of cycle averages
    """
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
    
    # Calculate overall metrics
    overall_avg = np.mean(cycle_averages) if cycle_averages else 0
    cycle_to_cycle_std = np.std(cycle_averages) if cycle_averages else 0
    scores_std = np.std(scores)if scores else 0
    return {
        'all_cycle_results': all_cycle_results,
        'overall_average': round(overall_avg,3),
        'cycle_to_cycle_std': round(cycle_to_cycle_std,3),
        'overall_scores_std': round(scores_std,2)
    }

# Example usage:
human_df = pd.read_csv(r'C:\Users\Valentin\Desktop\data_for_impact_analysis\data\inTouch\team_data\AlexeyM.txt')  # Load your human responses
llm_csv = r"C:\Users\Valentin\Desktop\data_for_impact_analysis\data\inTouch\outputs\ycloudml_analysis_results.csv"
change_requests = r"C:\Users\Valentin\Desktop\data_for_impact_analysis\data\inTouch\change_requests.txt"
results = calculate_alignment_scores_for_all_cycles(llm_csv, change_requests, human_df)
print(results)
