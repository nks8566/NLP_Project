import os
import json
import pandas as pd

def load_split(data_dir, split_type="random", split_name="train", use_processed=True):
    """
    data_dir: path to your data folder
    split_type: "random" or "media"
    split_name: "train", "val", or "test"
    use_processed: True -> use 'content', False -> use 'content_original'
    """

    # Paths
    json_dir = os.path.join(data_dir, "jsons")
    split_file = os.path.join(data_dir, "splits", split_type, f"{split_name}.tsv")

    # Load split file
    df = pd.read_csv(split_file, sep="\t")

    texts = []
    labels = []

    for _, row in df.iterrows():
        article_id = row["ID"]
        label = row["bias"]  # numeric label (0,1,2)

        json_path = os.path.join(json_dir, f"{article_id}.json")

        if not os.path.exists(json_path):
            continue  # skip missing files

        with open(json_path, "r", encoding="utf-8") as f:
            article = json.load(f)

        text = article["content"] if use_processed else article["content_original"]

        texts.append(text)
        labels.append(label)

    return texts, labels