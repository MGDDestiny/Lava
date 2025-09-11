import json
import pandas as pd
import os

"""
Utils to convert the results of the experiments to an excel file
"""

dataframes = []

for path in os.listdir('./pred/'):
    file_path = f'./pred/{path}/result.json'
    if not os.path.exists(file_path):
        continue
    with open(file_path, 'r') as file:
        data = json.load(file)

    column_order = [
        "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa",
        "musique", "dureader", "gov_report", "qmsum", "vcsum", "multi_news", "trec",
        "triviaqa", "samsum", "lsht",  "passage_count", "passage_retrieval_en", "passage_retrieval_zh",
        "lcc", "repobench-p"
    ]

    data_renamed = {
        "narrativeqa": data.get("narrativeqa", -1),
        "qasper": data.get("qasper", -1),
        "multifieldqa_en": data.get("multifieldqa_en", -1),
        "hotpotqa": data.get("hotpotqa", -1),
        "2wikimqa": data.get("2wikimqa", -1),
        "musique": data.get("musique", -1),
        "gov_report": data.get("gov_report", -1),
        "qmsum": data.get("qmsum", -1),
        "multi_news": data.get("multi_news", -1),
        "trec": data.get("trec", -1),
        "triviaqa": data.get("triviaqa", -1),
        "samsum": data.get("samsum", -1),
        "passage_count": data.get("passage_count", -1),
        "passage_retrieval_en": data.get("passage_retrieval_en", -1),
        "lcc": data.get("lcc", -1),
        "repobench-p": data.get("repobench-p", -1),
        "multifieldqa_zh": data.get("multifieldqa_zh", -1),
        "dureader": data.get("dureader", -1),
        "vcsum": data.get("vcsum", -1),
        "lsht": data.get("lsht", -1),
        "passage_retrieval_zh": data.get("passage_retrieval_zh", -1)
    }

    df = pd.DataFrame([data_renamed], columns=column_order, index=[path])
    dataframes.append(df)

result_df = pd.concat(dataframes)
output_path = './pred/combined_results.xlsx'
result_df.to_excel(output_path, index_label='Folder Name')

print(f"Excel file saved to {output_path}")
