from preprocess import preprocessing_docs
import pandas as pd
import numpy as np
import time

def get_data(csv_file):

	start = time.time()
    
    # Reading data
	incidents = pd.read_csv(csv_file)
	null_columns=incidents.columns[incidents.isnull().any()]
	null_cols = incidents[null_columns].isnull().sum()
	print('Total Rows - ',len(incidents.index))
	print(' ')
	print('null rows in data')
	print(' ')
	print(null_cols)
	print('----------------------------------------')




	# For Short_Description
	print('For Short_Description.............................')
	incidents_short_description = incidents.copy()
	incidents_short_description.dropna(subset=['short_description'], inplace=True)
	print('Line items in df ',incidents_short_description.describe().iloc[0,0])
	short_description = incidents_short_description['short_description'].values
	print('Total docs in Short_Description: ',len(short_description))

	# For Description
	print('For Description.............................')
	incidents_description = incidents.copy()
	incidents_description.dropna(subset=['description'], inplace=True)
	print('Line items in df ',incidents_description.describe().iloc[0,0])
	description = incidents_description['description'].values
	print('Total docs in description: ',len(description))

	# For close_notes
	print('For Close_Notes.............................')
	incidents_close_notes = incidents.copy()
	incidents_close_notes.dropna(subset=['close_notes'], inplace=True)
	print('Line items in df ',incidents_close_notes.describe().iloc[0,0])
	close_notes = incidents_close_notes['close_notes'].values
	print('Total docs in close_notes: ',len(close_notes))

	
	# For short_description_category_subcategory
	print('For short_description_category_subcategory.............................')
	incidents_short_description_category_subcategory = incidents.copy()

	incidents_short_description_category_subcategory.fillna({
	    'category': ' ',
	    'subcategory': ' ',
	    'short_description': ' '
	}, inplace=True)

	incidents_short_description_category_subcategory['short_description_category_subcategory'] = incidents_short_description_category_subcategory['short_description'] + ' ' + incidents_short_description_category_subcategory['category'] + ' ' + incidents_short_description_category_subcategory['subcategory']
	incidents_short_description_category_subcategory.dropna(subset=['short_description_category_subcategory'], inplace=True)
	short_description_category_subcategory = incidents_short_description_category_subcategory['short_description_category_subcategory'].values
	print('Total docs in short_description_category_subcategory: ',len(short_description_category_subcategory))

	
	# For description_category_subcategory
	print('For Description_category_subcategory.............................')
	incidents_description_category_subcategory = incidents.copy()

	incidents_description_category_subcategory.fillna({
	    'category': ' ',
	    'subcategory': ' ',
	    'description': ' '
	}, inplace=True)

	incidents_description_category_subcategory['description_category_subcategory'] = incidents_description_category_subcategory['description'] + ' ' + incidents_description_category_subcategory['category'] + ' ' + incidents_description_category_subcategory['subcategory']
	incidents_description_category_subcategory.dropna(subset=['description_category_subcategory'], inplace=True)
	description_category_subcategory = incidents_description_category_subcategory['description_category_subcategory'].values
	print('Total docs in description_category_subcategory: ',len(description_category_subcategory))

	
	# For close_notes_category_subcategory
	print('For Close_Notes_category_subcategory.............................')
	incidents_close_notes_category_subcategory = incidents.copy()

	incidents_close_notes_category_subcategory.fillna({
	    'category': ' ',
	    'subcategory': ' ',
	    'close_notes': ' '
	}, inplace=True)

	incidents_close_notes_category_subcategory['close_notes_category_subcategory'] = incidents_close_notes_category_subcategory['close_notes'] + ' ' + incidents_close_notes_category_subcategory['category'] + ' ' + incidents_close_notes_category_subcategory['subcategory']
	incidents_close_notes_category_subcategory.dropna(subset=['close_notes_category_subcategory'], inplace=True)
	close_notes_category_subcategory = incidents_close_notes_category_subcategory['close_notes_category_subcategory'].values
	print('Total docs in close_notes_category_subcategory: ',len(close_notes_category_subcategory))

	print('----------------------------------------')
	print('Total docs in Short_Description: ',len(short_description))
	print('Total docs in description: ',len(description))
	print('Total docs in close_notes: ',len(close_notes))
	print('Total docs in short_description_category_subcategory: ',len(short_description_category_subcategory))
	print('Total docs in description_category_subcategory: ',len(description_category_subcategory))
	print('Total docs in close_notes_category_subcategory: ',len(close_notes_category_subcategory))

	print('----------------------------------------')
	end = time.time()
	print('Time taken = ', end-start)

	return short_description, description, close_notes, short_description_category_subcategory, description_category_subcategory, close_notes_category_subcategory