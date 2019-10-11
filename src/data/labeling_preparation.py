import json
import os
import pickle as pkl
import shutil
import zipfile
from datetime import datetime
from typing import Optional

from newsplease import NewsArticle
from tqdm import tqdm

from settings import PROJECT_PATH

TEXTS_PATH = os.path.join(PROJECT_PATH, 'data', 'texts')
UNZIPPED_DESTINATION = "unzipped"
DEST_ABS_PATH = os.path.join(TEXTS_PATH, UNZIPPED_DESTINATION)


def datetime_to_str(date: Optional[datetime]) -> Optional[str]:
    return date.strftime("%m/%d/%Y, %H:%M:%S") if date else ""


def read_single_article_group(path: str) -> NewsArticle.NewsArticle:
    """
    Read NewsArticle object from dumped pickle file
    :param path: path to pickle
    :return: NewsArticle object
    """
    try:
        with open(path, 'rb') as fp:
            article = pkl.load(fp)
            return article
    except (OSError, IOError) as _:
        print("Could not read ", path, " Skipping")


def restructurize_article_object(article_object: NewsArticle.NewsArticle, month: str,
                                 group_number: str) -> Optional[dict]:
    """
    Takes the NewsArticle object and converts it to doccano-compatibile JSON Lite record.
    Namely, "text" key must contain text of article and "meta" anything we want to attach to a
    record
    :param article_object: previously read NewsArticle object
    :param month: currently processed month that will be attached to metadata
    :param group_number: id within given month, also attached to metadata
    :return: dict, restructurised article_object
    """
    data_dict = article_object.get_dict()
    article_text = data_dict.pop('text')

    # sometimes a page was unavailable or link was no longer valid and in result, text is None.
    # some of these pages are available now but they constitute very small subset (ca.
    # 2300/65000) are I decided to just skip them
    if not article_text:
        return None

    meta_dict = {
        'date_publish': datetime_to_str(data_dict['date_publish']),
        'date_download': datetime_to_str(data_dict['date_download']),
        'date_modify': datetime_to_str(data_dict['date_modify']),
        'month': month,
        'id': group_number,
        'url': data_dict['url'],
        'source_domain': data_dict['source_domain'],
        'title': data_dict['title']
    }

    return {'text': article_text, 'meta': meta_dict}


def unzip_articles_text(destination=DEST_ABS_PATH, forced=False):
    if os.path.exists(destination):
        if forced:
            shutil.rmtree(destination)
        else:
            print("Nothing to be done. Destination seems populated")
            return

    for _, _, files in os.walk(TEXTS_PATH):
        for month in tqdm(files, desc="Unzipping text articles"):
            month_absolute_path = os.path.join(TEXTS_PATH, month)
            with zipfile.ZipFile(month_absolute_path, 'r') as fp:
                fp.extractall(destination)


def prepare_doccano_json(destination_filepath="output.json"):
    failure_number = 0

    with open(destination_filepath, 'w') as fp:
        for _, subfolders, _ in os.walk(DEST_ABS_PATH):
            for month in sorted(subfolders):
                month_destination = os.path.join(DEST_ABS_PATH, month)
                for _, _, groups in os.walk(month_destination):
                    for article_group in tqdm(groups, desc="Packing month " + month):
                        article_path = os.path.join(month_destination, article_group)
                        art_obj = read_single_article_group(article_path)
                        doccano_json_record = restructurize_article_object(art_obj, month,
                                                                           article_group)
                        if not doccano_json_record:
                            failure_number += 1
                            continue
                        json.dump(doccano_json_record, fp)
                        fp.write("\n")  # doccano requires each record be separated by line
    print("Done with " + str(failure_number) + " failures")


if __name__ == "__main__":
    unzip_articles_text(forced=True)
    prepare_doccano_json()
