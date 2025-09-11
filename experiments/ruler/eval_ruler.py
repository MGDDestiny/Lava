import os
import json
import argparse
import numpy as np

from metrics import (
    string_match_all
)

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default=None)
    return parser.parse_args(args)

if __name__ == '__main__':
    args = parse_args()
    
    dataset_list = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
            "niah_multiquery", "niah_multivalue", "cwe", "fwe", "vt"]
    
    
    for dataset in dataset_list:
        print(f"Evaluating {dataset}")
        scores = dict()
        
        try:
            for f in os.listdir(os.path.join(args.results_dir, dataset)):
                if 'metrics.json' in f:
                    continue
                args.method = f
                args.dataset = dataset
                args.eval_file = os.path.join(args.results_dir,dataset,f)
                predictions, answers, lengths = [], [], []
                # dataset = filename.split('.')[0]
                with open(args.eval_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            predictions.append(data["pred"])
                            answers.append(data["answers"])
                            if "length" in data:
                                lengths.append(data["length"])
                        except:
                            print("error")
                
                score = string_match_all(predictions, answers)
                scores[args.method] = score
                
                output_dir = os.path.dirname(args.eval_file)
                with open(os.path.join(output_dir, "metrics.json"), "w") as f:
                    json.dump(scores, f, ensure_ascii=False, indent=4)
                # print(f"dataset {args.dataset} method {args.method} scores {scores}")
        except:
            print(f"Error evaluating {dataset}")
            continue