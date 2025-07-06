import os
import re
import pandas as pd
import time
from mistralai import Mistral

# Supporting functions for impact analysis
# The usage is mainly to run_full_analysis and to_the_forms

def load_model_data(model_path):
    """Load model data from a file."""
    with open(model_path, "r", encoding="utf-8") as f:
        return f.read()

def load_change_requests(change_requests_path):
    """Load change requests from a file."""
    with open(change_requests_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def extract_suggestions(text):
    """Extract final answer block from response."""
    code_blocks = re.findall(r"&&&(.*?)&&&", text, re.DOTALL)
    return code_blocks[0].strip() if code_blocks else ""

def create_prompt(model_data, change_request, proj_description):

    """Create the prompt for impact analysis."""

    return f"""

    ## Role
    You are the system engineer in the start up team.
    ## Context
    {proj_description}
    You have to assess the potential consequences of a change or modification to a system i.e. conduct impact analysis. The team had built the model of the product they are developing. It is presented below:
    ```
    {model_data}
    ```
    The specific incoming change that to be analyzed is: `{change_request}`.
    ## Instruction
    Complete your task in phases. Complete each phase and then proceed to the next:
    1. Phase 1: Identify impact:
    List all of the components that is directly influenced by the incoming change along with the reasoning.
    - A component is said to be **directly influenced (impacted)** by a change if: "It must undergo a modification as a first-order consequence of the proposed change, without requiring propagation through intermediary components, i.e. as secondary or cascading effect."
    2. Phase 2: Self-evaluation and verification
    - Cross-check each pair: "Is this impact logically consistent with the model’s architecture? Explain in 10 words."
    3. Phase 3: Final output print
    - Using verified pairs create the final output with the specific format described below:
    ```
    &&&
    Component: <component_name>
    Reasoning: <reasoning_text>
    
    Component: <component_name>
    Reasoning: <reasoning_text>
   
    Component: <component_name>
    Reasoning: <reasoning_text>

    Component: <component_name>
    Reasoning: <reasoning_text>

    ...
    &&&
    ```
    ## Constraints
    - Use **exact component names** from the model.
    - List only the components in your analysis. Definition: Component is the smallest functional or structural unit of a system. In could be internal and external.)



    """

def parse_response(response_text, change_request_id, cycle_number):
    """Parse the API response into a DataFrame."""
    data_string = extract_suggestions(response_text)
    if not data_string:
        print(f"No valid output for change request {change_request_id}")
        return pd.DataFrame()

    entries = data_string.split('\n\n')
    data = []

    for entry in entries:
        component_match = re.search(r'Component:\s*(.*)', entry)
        reasoning_match = re.search(r'Reasoning:\s*(.*)', entry, re.DOTALL)
        if component_match and reasoning_match:
            component = component_match.group(1).strip()
            reasoning = reasoning_match.group(1).strip().replace('\n', ' ')
            data.append({
                "Component": component, 
                "Reasoning": reasoning,
                "Change": change_request_id,
                "Cycle": cycle_number
            })

    return pd.DataFrame(data)

def get_api_response(client, agent_id, prompt):
    """Get response from the Mistral API."""
    return client.agents.complete(
        agent_id=agent_id,
        messages=[{"role": "user", "content": prompt}]
    )

def process_single_change_request(model_data, change_request, change_request_id, cycle_number, project_description, client=None, agent_id=None, manual_response=None):
    """
    Process a single change request either via API or with manual response.
    
    Args:
        model_data: The model data as string
        change_request: The change request text
        change_request_id: Numeric ID for the change request
        cycle_number: Current cycle number
        client: Mistral client (required if using API)
        agent_id: Agent ID (required if using API)
        manual_response: Pre-generated response text (if not using API)
        project_description: Project description
    
    Returns:
        pd.DataFrame with impact analysis results
    """
    prompt = create_prompt(model_data, change_request, project_description)
    
    if manual_response:
        response_text = manual_response
    else:
        if not client or not agent_id:
            raise ValueError("Both client and agent_id must be provided when not using manual_response")
        chat_response = get_api_response(client, agent_id, prompt)
        response_text = chat_response.choices[0].message.content
    
    return parse_response(response_text, change_request_id, cycle_number)

def process_all_change_requests(model_path, change_requests_path, cycle_number, project_description, client=None, agent_id=None, manual_responses=None, pause=0):
    """
    Process all change requests either via API or with manual responses.
    
    Args:
        model_path: Path to model data file
        change_requests_path: Path to change requests file
        cycle_number: Current cycle number
        client: Mistral client (optional if using manual_responses)
        agent_id: Agent ID (optional if using manual_responses)
        manual_responses: List of pre-generated responses (optional)
        pause: Seconds to pause between requests
        
    Returns:
        Combined DataFrame with all impacts
    """
    model_data = load_model_data(model_path)
    change_requests = load_change_requests(change_requests_path)
    
    if manual_responses and len(manual_responses) != len(change_requests):
        raise ValueError("Number of manual responses must match number of change requests")
    
    combined_df = pd.DataFrame()
    
    for idx, change_request in enumerate(change_requests, start=1):
        print(f"Processing request {idx}/{len(change_requests)}...")
        
        manual_response = manual_responses[idx-1] if manual_responses else None
        df = process_single_change_request(
            model_data=model_data,
            change_request=change_request,
            change_request_id=idx,
            cycle_number=cycle_number,
            project_description=project_description,
            client=client,
            agent_id=agent_id,
            manual_response=manual_response
        )
        
        if not df.empty:
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        
        time.sleep(pause)
    
    return combined_df

def run_full_analysis(model_path, change_requests_path, project_description,num_cycles=1, pause_between_cycles=3, pause_between_requests=5, api_key=None, agent_id=None):
    """
    Run full analysis with multiple cycles using API calls.
    
    Args:
        model_path: Path to model data file
        change_requests_path: Path to change requests file
        num_cycles: Number of cycles to run
        pause_between_cycles: Seconds between cycles
        pause_between_requests: Seconds between requests
        api_key: Mistral API key
        agent_id: Agent ID
        
    Returns:
        Combined DataFrame with all impacts from all cycles
    """
    client = Mistral(api_key=api_key) if api_key else None
    all_runs_df = pd.DataFrame()

    for cycle in range(1, num_cycles + 1):
        print(f"\n=== Running cycle {cycle}/{num_cycles} ===")
        
        df = process_all_change_requests(
            model_path=model_path,
            change_requests_path=change_requests_path,
            cycle_number=cycle,
            project_description=project_description,
            client=client,
            agent_id=agent_id,
            pause=pause_between_requests
        )
        
        all_runs_df = pd.concat([all_runs_df, df], ignore_index=True)
        
        if cycle < num_cycles:
            print(f"Waiting {pause_between_cycles} seconds before next cycle...")
            time.sleep(pause_between_cycles)
    
    return all_runs_df

def to_the_forms(df, change_requests_path, change_column="Change"):
    """
    Maps change IDs in a DataFrame column to their corresponding text descriptions
    using a mapping file.
    
    Args:
        df: Input DataFrame
        change_column: Name of the column containing change IDs (default: "Change")
        change_requests_path: Path to the text file containing change descriptions 
                            (one per line, line number = change ID)
    
    Returns:
        DataFrame with the change IDs replaced by their text descriptions
    """
    # Create a copy of the input DataFrame
    df_copy = df.copy()
    
    # Read change descriptions from file
    with open(change_requests_path, "r", encoding="utf-8") as f:
        change_text = [line.strip() for line in f if line.strip()]
    
    # Create mapping dictionary (ID -> text)
    change_map = {idx+1: text for idx, text in enumerate(change_text)}
    
    # Map IDs to text in the DataFrame column
    df_copy[change_column] = df_copy[change_column].map(change_map)
    
    return df_copy.to_csv(os.path.join("outputs", "to-the-form.csv"), index = False)


def add_change_id_column(df, change_column_name, file_path='./change_requests.txt'):
    """
    Adds a change_id column to a DataFrame by matching change descriptions to IDs from a text file.
    
    Args:
        df: Input DataFrame
        change_column_name: Name of the column containing change descriptions
        file_path: Path to the text file containing change requests (one per line)
        
    Returns:
        DataFrame with the added change_id column
    """
    # Read the change requests from the text file
    with open(file_path, 'r') as file:
        change_requests = [line.strip() for line in file.readlines() if line.strip()]

    # Create a dictionary mapping change descriptions to their IDs (line numbers)
    change_to_id = {change: idx+1 for idx, change in enumerate(change_requests)}

    # Function to find the matching change ID
    def find_change_id(change_description):
        # Try exact match first
        if change_description in change_to_id:
            return change_to_id[change_description]
        
        # If not found exactly, try to find the most similar one
        for change in change_to_id:
            if change in change_description or change_description in change:
                return change_to_id[change]
        
        # If still not found, return None
        return None

    # Apply the function to create the new change_id column
    df['change_id'] = df[change_column_name].apply(find_change_id)

    # Print warnings for unmatched changes
    if df['change_id'].isnull().any():
        unmatched_count = df['change_id'].isnull().sum()
        print(f"Warning: Could not match {unmatched_count} change requests to IDs")
        print("Unmatched changes:")
        print(df[df['change_id'].isnull()][change_column_name])

def read_numbered_txt_files(folder_path):
    """
    Reads numbered .txt files from a folder (starting from 1) and returns their contents as a list.
    
    Args:
        folder_path (str): Path to the folder containing the numbered .txt files
        
    Returns:
        list: A list where each element contains the content of a file (index 0 = file 1.txt)
        
    Raises:
        ValueError: If the folder doesn't exist or doesn't contain numbered files starting from 1
    """
    # Check if folder exists
    if not os.path.isdir(folder_path):
        raise ValueError(f"Folder not found: {folder_path}")
    
    contents = []
    file_number = 1
    
    while True:
        file_path = os.path.join(folder_path, f"{file_number}.txt")
        
        # Stop when we can't find the next numbered file
        if not os.path.exists(file_path):
            # Check if we found at least one file
            if file_number == 1:
                raise ValueError(f"No numbered .txt files found in {folder_path}")
            break
            
        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as f:
            contents.append(f.read().strip())
            
        file_number += 1
    
    return contents

def create_prompt_files(change_requests_path, export_json_path,project_description, output_folder="to-the-chat"):
    """
    Creates prompt files for each change request combining with model data.
    
    Args:
        change_requests_path: Path to change_requests.txt
        export_json_path: Path to export.json with model data
        project_description: Project description
        output_folder: Folder to save the prompt files (default: "to-the-chat")
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    change_requests = load_change_requests(change_requests_path)
    model_data = load_model_data(export_json_path)
    
    # Create a prompt for each change request
    for idx, change_request in enumerate(change_requests, start=1):
        # Create the prompt text
        prompt = create_prompt(model_data, change_request, project_description)
        
        output_path = os.path.join(output_folder, f"{idx}.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(prompt)
        
        print(f"Created prompt file: {output_path}")

def calculate_alignment_scores(change_requests_path, df1, df2):
    """
    Calculate alignment scores for all change requests and return scores with average.
    
    Args:
        change_requests: List of change request IDs to evaluate
        df1: LLM dataframe (containing 'Change' and 'Component' columns)
        df2: Engineer dataframe (containing 'Change' and 'Component' columns)
        
    Returns:
        tuple: (list_of_scores, average_score)
    """
    alignment_scores = []
    change_requests = load_change_requests(change_requests_path)
    
    for idx,change_val in enumerate(change_requests):
        # Get components for this change value in both DataFrames
        comp1 = df1[df1['Change'] == idx+1]['Component']
        comp2 = df2[df2['Change'] == idx+1]['Component']
        
        # Calculate alignment score
        common_components = set(comp1) & set(comp2)
        len_common = len(common_components)
        len_eng = len(comp2) if len(comp2) > 0 else 1  # Avoid division by zero
        alignment_score = len_common / len_eng
        
        alignment_scores.append(alignment_score)
    
    # Calculate average, handling empty list case
    average_score = sum(alignment_scores)/len(alignment_scores) if alignment_scores else 0
    
    return alignment_scores, average_score

def calculate_novelty_scores(df1, df2):
    """
    Calculate novelty scores for each change where:
    Novelty Score = (Number of new relevant components) / (Total relevant components in df1 for that change)
    
    Returns a DataFrame with Change and Novelty_Score columns
    """
    # Filter relevant components in df1 (case insensitive)
    df1_relevant = df1[df1['Relevant'].str.lower() == 'yes'].copy()
    df1_relevant['Component_lower'] = df1_relevant['Component'].str.lower()
    
    # Prepare df2 components (case insensitive)
    df2 = df2.copy()
    df2['Component_lower'] = df2['Component'].str.lower()
    
    # Get common changes between both dataframes
    common_changes = set(df1['Change']).intersection(set(df2['Change']))
    
    novelty_scores = []
    
    for change in common_changes:
        # Get components for this change in both dataframes
        df1_components = set(df1_relevant[df1_relevant['Change'] == change]['Component_lower'])
        df2_components = set(df2[df2['Change'] == change]['Component_lower'])
        
        # Calculate novelty metrics
        total_components = len(df1_components)
        new_components = len(df1_components - df2_components)
        
        if total_components > 0:  # Avoid division by zero
            novelty_score = new_components / total_components
        else:
            novelty_score = 0.0
        
        novelty_scores.append(novelty_score)
    avg_novelty = sum(novelty_scores)/len(novelty_scores) if novelty_scores else 0
    return novelty_scores, avg_novelty

def extract_relevacy(survey_path):
    """
    Creates dataframe relevance column from survey row 
    """
    df_n = pd.read_csv(filepath_or_buffer=survey_path, sep = '\t', header = None)
    df_n = df_n.drop(df_n.columns[[0,1]], axis=1).T  # Drop columns at positions 0, 1
    df_n.columns = ['Relevant']
    df_n.index = range(0, len(df_n))
    return df_n