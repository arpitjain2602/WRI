from preprocess.arpit_v2 import *
from preprocess.preprocess_v2 import *
from preprocess.preprocess_v2 import preprocess
import os
import inspect
import time
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import pickle
from models.LDA_multi_level import lda_model_multi_level
from models.LDA_single_level import lda_model_single_level
print('------------------------------------------------------')
print('- Imports Done')


# DATA
# Note that raw docs is a numpy array. 
# Example element is: 
# 'Logical Disk Free Space is low, Description: The disk C: on computer sjcphxstg02.strykercorp.com is running out of disk space. The values that exceeded the thre'
# data_file_string = 'short_description.pkl'
data_file_string = 'data.pkl'
data_file = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))),'data',data_file_string)
raw_docs = pickle.load(open(data_file,'rb'))
print('- Imported Data')

# PRE-PROCESSING
preprocess_steps_and_order = {
	'make_lowercase': [True],
	'punctuation_removal':[True],
	'whitespace_removal': [True],
	'store_alphanumeric': [False],
	'pos_removal_nltk': [True],
	'tokenization_nltk': [False],
	'lemmatization_tokenization_spacy': [True],
	'stopwords_removal_nltk': [True],
	'stopwords_removal_spacy': [False],
	'make_bigrams_gensim':[True, {'make_bigrams_gensim': True, 'bigrams_min_count': 10, 'bigrams_threshold': 10}],
	'make_trigrams_gensim':[True, {'make_trigrams_gensim': True, 'trigrams_min_count': 10, 'trigrams_threshold': 10}],
	'min_max_length_removal':[False, {'min_max_length_removal': False, 'mmlr_min_len': 3, 'mmlr_max_len': 50, 'mmlr_deacc': False}]
	}

preprocess_functions = {
	'make_lowercase': make_lowercase,
	'punctuation_removal': punctuation_removal,
	'whitespace_removal': whitespace_removal,
	'store_alphanumeric': store_alphanumeric,
	'pos_removal_nltk': pos_removal_nltk,
	'tokenization_nltk': tokenization_nltk,
	'lemmatization_tokenization_spacy': lemmatization_tokenization_spacy,
	'stopwords_removal_nltk': stopwords_removal_nltk,
	'stopwords_removal_spacy': stopwords_removal_spacy,
	'make_bigrams_gensim': make_bigrams_gensim,
	'make_trigrams_gensim': make_trigrams_gensim,
	'min_max_length_removal': min_max_length_removal
	}


# MODELS
models_dict = {
	'LDA_single_level': lda_model_single_level,
	'LDA_multi_level': lda_model_multi_level,
}


# IDEA - Dynamically increase the topics based on the coherence score calculated each time

# SPECIFICATIONS
specifications = {
	# 'model':'LDA_single_level', # Can be LDA_multi_level
	'level':2,
	'num_topics_list_level_1':[5],
	'num_topics_list_level_2':[1,2,3,4,5],
	'num_topics_list_level_3':[1,2,3,4,5],
	'coherence':'c_v',
	'need_best_topic': True,
	'model_selection_metric':'coherence', # or 'perplexity',
	'debug':False,
}

print('*****************************************************')
print('- Starting preprocessing')
dictionary, corpus, doc_list = preprocess(
								raw_docs = raw_docs, 
								preprocess_functions = preprocess_functions, 
								preprocess_steps_and_order = preprocess_steps_and_order, 
								debug=specifications['debug'])

print('*****************************************************')
print('- Starting model training')
lda_dict = lda_model_single_level(
					dictionary = dictionary,
					corpus = corpus,
					doc_list = doc_list,
					num_topics_list_level_1 = specifications['num_topics_list_level_1'], 
					coherence = specifications['coherence'],
					debug = specifications['debug'],
					need_best_topic = specifications['need_best_topic'],
					model_selection_metric = specifications['model_selection_metric']
					)

print(lda_dict['coherence_score'])

# lda_dict = lda_model_multi_level(
# 					coherence = specifications['coherence'],
# 					debug = specifications['debug'],
# 					need_best_topic = specifications['need_best_topic'],
# 					model_selection_metric = specifications['model_selection_metric'],
# 					num_topics_list_level_2 = specifications['num_topics_list_level_2'], 
# 					level = specifications['level'],
# 					num_topics_list_level_1 = specifications['num_topics_list_level_1'], 
# 					raw_docs = raw_docs)

print('*****************************************************')
print('- Starting model training')
lda_level_1, lda_level_2 = lda_model_multi_level(
					level = specifications['level'],
					dictionary = dictionary,
					corpus = corpus,
					doc_list = doc_list,
					coherence = specifications['coherence'],
					debug = specifications['debug'],
					need_best_topic = specifications['need_best_topic'],
					model_selection_metric = specifications['model_selection_metric'],
					num_topics_list_level_1 = specifications['num_topics_list_level_1'], 
					num_topics_list_level_2 = specifications['num_topics_list_level_2'], 
					)