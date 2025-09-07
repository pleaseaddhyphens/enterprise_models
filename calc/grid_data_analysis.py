import os
import pandas as pd
from utils import calculate_f_scores
import json

PROJECT_PATH = r"C:\Users\Valentin\Desktop\data_for_impact_analysis\data\FiberPipe"

CHANGE_REQUESTS_PATH = os.path.join(PROJECT_PATH, "change_requests.txt")
OUTPUTS_DIR = os.path.join(PROJECT_PATH, "outputs")
TEAM_DATA = os.path.join(PROJECT_PATH, "team_data")


def calculate_all_grid():
    results = {}
    # List all output files
    output_files = [f for f in os.listdir(OUTPUTS_DIR) if f.endswith('.csv')]
    # List all participant files (exclude _survey)
    participant_files = [f for f in os.listdir(TEAM_DATA) if not f.endswith('_survey.txt') and f.endswith('.txt')]

    for output_file in output_files:
        output_path = os.path.join(OUTPUTS_DIR, output_file)
        df1 = pd.read_csv(output_path)
        participant_scores = {}
        avg_f_scores = []
        avg_recall_scores = []
        for participant_file in participant_files:
            participant_path = os.path.join(TEAM_DATA, participant_file)
            df2 = pd.read_csv(participant_path)
            scores, avg_score, recall = calculate_f_scores(CHANGE_REQUESTS_PATH, df1, df2)
            participant_name = participant_file.replace('.txt', '')
            participant_scores[participant_name] = round(avg_score,2)
            avg_f_scores.append(round(avg_score,2))
            avg_recall_scores.append(round(recall,2))
        # Add average for the team
        participant_scores['average_f'] = sum(avg_f_scores) / len(avg_f_scores) if avg_f_scores else 0
        participant_scores['average_recall'] = sum(avg_recall_scores) / len(avg_recall_scores) if avg_recall_scores else 0
        results[output_file] = participant_scores
    return results

all_scores = calculate_all_grid()
print(json.dumps(all_scores, indent=4))

