"""Reads PDF to text

Credit to `Kulsoom Abdullah`
"""
import glob
import os
import re
import pandas as pd
from tika import parser


def read_document(filename):
    file_data = parser.from_file(filename)
    content = file_data['content']
    try:
        return re.sub("(\\n)+", " ", content)
    except (AttributeError, TypeError):
        return content


def read_folder(folder_path):
    """Reads all pdf from a folder"""
    files = glob.iglob(os.path.join(folder_path, "*.pdf"))
    for file in files:
        name = os.path.basename(file)
        yield name.replace(".pdf", ""), read_document(os.path.join(folder_path, file))

def pdf_df(folder_path):
    """Puts data from pdf folder into a pandas dataframe"""
    return pd.DataFrame(
        [[name, text] for name, text in read_folder(folder_path)],
        columns=['policy_name', 'policy_text'])

