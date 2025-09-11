
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import json
import glob
import argparse

# FOLDER_PATH = "./results_needle/results/mistral2_7b_pyramidkv_128_run/"
# MODEL_NAME = "mistral_2_7b"
PRETRAINED_LEN=30000


def main(folder_path, model_name, output_path):
    # Path to the directory containing JSON results
    if("/" in folder_path):
        model_name = folder_path.split("/")[-2]
    else: pass
    print("model_name = %s" % model_name)
    # Using glob to find all json files in the directory
    json_files = glob.glob(f"{folder_path}/*.json")
    # import ipdb; ipdb.set_trace()

    # List to hold the data
    data = []

    # Iterating through each file and extract the 3 columns we need
    
    for file in json_files:
        with open(file, 'r') as f:
            json_data = json.load(f)
            # Extracting the required fields
            
            try:
                document_depth = json_data.get("depth_percent", None)
                context_length = json_data.get("context_length", None)
            except:
                import pdb
                pdb.set_trace()
            # score = json_data.get("score", None)
            model_response = json_data.get("model_response", None).lower()
            needle = json_data.get("needle", None).lower()
            expected_answer = "eat a sandwich and sit in Dolores Park on a sunny day".lower().split()
            model_response = model_response.replace(".", " ")
            score = len(set(model_response.split()).intersection(set(expected_answer))) / len(set(expected_answer))
            # Appending to the list
            data.append({
                "Document Depth": document_depth,
                "Context Length": context_length,
                "Score": score
            })

    # Creating a DataFrame
    df = pd.DataFrame(data)
    locations = list(df["Context Length"].unique())
    locations.sort()
    for li, l in enumerate(locations):
        if(l > PRETRAINED_LEN): break
    pretrained_len = li

    print(df.head())
    print("Overall score %.5f" % df["Score"].mean())

    pivot_table = pd.pivot_table(df, values='Score', index=['Document Depth', 'Context Length'], aggfunc='mean').reset_index() # This will aggregate
    pivot_table = pivot_table.pivot(index="Document Depth", columns="Context Length", values="Score") # This will turn into a proper pivot
    pivot_table.iloc[:5, :5]

    # Create a custom colormap. Go to https://coolors.co/ and pick cool colors
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])

    # Create the heatmap with better aesthetics
    f = plt.figure(figsize=(38, 8))  # Can adjust these dimensions as needed
    heatmap = sns.heatmap(
        pivot_table,
        vmin=0, vmax=1,
        cmap=cmap,
        cbar_kws={'label': 'Score'},
        linewidths=0.5,  # Adjust the thickness of the grid lines here
        linecolor='grey',  # Set the color of the grid lines
        linestyle='--'
    )


    # More aesthetics
    model_name_ = model_name
    plt.title(f'Pressure Testing {model_name_} \nFact Retrieval Across Context Lengths ("Needle In A HayStack")', fontsize=18)  # Adds a title
    plt.xlabel('Token Limit', fontsize=18)  # X-axis label
    plt.ylabel('Depth Percent', fontsize=18)  # Y-axis label
    plt.xticks(rotation=45, fontsize=18)  # Rotates the x-axis labels to prevent overlap
    plt.yticks(rotation=0, fontsize=18)  # Ensures the y-axis labels are horizontal
    plt.tight_layout()  # Fits everything neatly into the figure area

    # Add a vertical line at the desired column index
    plt.axvline(x=pretrained_len + 0.8, color='white', linestyle='--', linewidth=4)

    save_path = f"{output_path}.png"
    print("saving at %s" % save_path)
    plt.savefig(save_path, dpi=150)

    return df["Score"].mean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', type=str, default=None, help='folder')
    parser.add_argument('--model_name', type=str, default=None, help='name of model')
    args = parser.parse_args()
    args.output_path = args.folder_path
    main(args.folder_path, args.model_name, args.output_path)