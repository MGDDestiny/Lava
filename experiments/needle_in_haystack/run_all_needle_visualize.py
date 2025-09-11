from run_needle_visulize import main
import argparse
import os
import json
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", type=str, default="")
    args = parser.parse_args()
    
    scores = dict()
    for folder in os.listdir(args.results_path):
        try:
            result_path = os.path.join(args.results_path, folder, "results")
            scores[folder] = main(folder_path=result_path, model_name=None, output_path=result_path)
        except:
            pass
    
    with open(f"{args.results_path}/all_needle_scores.json", "w") as f:
        json.dump(scores, f, indent=4)
