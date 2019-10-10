import csv
import warnings
from typing import Dict
import json
import jsonlines as jsl
from tqdm import tqdm


def read_labels_dict(labels_filepath: str) -> Dict:
    """
    In file that contains the annotated articles, all annotations are encoded by user/label
    id to save up space etc. To provide readable and similar format to gold-standard we need to
    decode these id to text, for example "1" -> "positive"
    :param labels_filepath: filepath to doccano label file
    :return: dict with mapping similar to description above
    """
    labels_dict = {}

    with open(labels_filepath, "r") as f:
        labels = json.load(f)

    for json_obj in labels:
        doccano_id = json_obj["id"]
        text = json_obj["text"]
        labels_dict[doccano_id] = text
    return labels_dict


def convert_labelled_data(annotation_filepath: str, labels_filepath: str,
                          output_filepath: str) -> None:
    """
    convert labelled data in json format to csv format like in gold-standard
    """
    labels_mapping = read_labels_dict(labels_filepath)

    with open(output_filepath, "w") as fp_writer:
        writer = csv.writer(fp_writer)
        writer.writerow(["ids", "month", "class", "title"]) # write column titles
        with jsl.Reader(open(annotation_filepath, "r")) as reader:
            for json_obj in tqdm(reader):
                annotations = json_obj['annotations']
                if not annotations:
                    continue

                metadata = json_obj['meta']

                group_id = str(int(metadata["id"].split(".")[0]))  # convert "0001.pkl" -> "1"
                month = metadata["month"]
                title = metadata["title"]

                if len(annotations) > 1:
                    warnings.warn(
                        "Found multilabeled annotation in month " + month + "id " + group_id)
                    # there is a posibility there are up to 5 such annotations where we were
                    # playing with project's settings
                label = labels_mapping[annotations[0]['label']]
                writer.writerow([group_id, month, label, title])


if __name__ == "__main__":
    convert_labelled_data("labelled.json", "positive_negative_label_mapping.json", "converted.csv")
