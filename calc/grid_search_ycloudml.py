import os
import itertools
import pandas as pd
from ycloudml_config import (
    YANDEX_CONFIG, CONTEXT_PATHS, CHANGE_REQUESTS_PATH, PROJECT_DESCRIPTION, ANALYSIS_CONFIG, OUTPUT_CONFIG, PARAM_GRID
)
from signle_run_analysis import run_full_analysis_ycloudml


def grid_search_ycloudml():
    # Prepare parameter grid
    param_names = list(PARAM_GRID.keys())
    param_values = [PARAM_GRID[name] for name in param_names]
    combinations = list(itertools.product(*param_values))

    results = []
    for combo in combinations:
        params = dict(zip(param_names, combo))
        model_name = params["model_name"]
        context_type = params["context_type"]
        model_path = CONTEXT_PATHS[context_type]

        print(f"\n=== Running grid search for: model_name={model_name}, context_type={context_type} ===")

        # Run analysis
        df = run_full_analysis_ycloudml(
            model_path=model_path,
            change_requests_path=CHANGE_REQUESTS_PATH,
            project_description=PROJECT_DESCRIPTION,
            folder_id=YANDEX_CONFIG["folder_id"],
            auth_token=YANDEX_CONFIG["auth_token"],
            model_name=model_name,
            temperature=YANDEX_CONFIG["temperature"],
            num_cycles=ANALYSIS_CONFIG["num_cycles"],
            pause_between_cycles=ANALYSIS_CONFIG["pause_between_cycles"],
            pause_between_requests=ANALYSIS_CONFIG["pause_between_requests"]
        )

        # Save results
        out_dir = OUTPUT_CONFIG["outputs_dir"]
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(
            out_dir,
            f"{context_type}_{model_name}_t{YANDEX_CONFIG['temperature']}_cyc{ANALYSIS_CONFIG['num_cycles']}.csv"
        )
        df.to_csv(out_csv, index=False)
        print(f"Saved results to {out_csv}")
        results.append({"params": params, "csv": out_csv, "num_impacts": len(df)})


if __name__ == "__main__":
    grid_search_ycloudml() 