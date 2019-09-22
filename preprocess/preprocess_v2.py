from preprocess.arpit_v2 import *
import time
import warnings
warnings.filterwarnings('ignore')
import pickle

'''
Order of preprocessing

make_lowercase
punctuation_removal
whitespace_removal
X store_alphanumeric
pos_removal_nltk
X tokenization_nltk
lemmatization_tokenization_spacy
stopwords_removal_nltk
stopwords_removal_spacy
make_bigrams_gensim
make_trigrams_gensim
X min_max_length_removal
X store_alphanumeric

The one with crosses shouldn't be done. Alphanumeric are not being deleted
by lemmatization-tokenization.

'''
global sample_to_print
sample_to_print = 3
def get_steps_list(preprocess_steps_and_order, debug=False):
	steps_list = []
	for key, value in preprocess_steps_and_order.items():
		if (debug):
			print(key, value[0])
		if (value[0]):
			steps_list.append(key)

	return steps_list


def preprocess(raw_docs, preprocess_functions, preprocess_steps_and_order, debug=False):
	'''
	- raw_docs:# Note that raw docs is a numpy array. 
		# Example element is: 
		# 'Logical Disk Free Space is low, Description: The disk C: on computer sjcphxstg02.strykercorp.com is running out of disk space. The values that exceeded the thre'
	- preprocess_steps_and_order: dictionary containg the order and functions to be cleaned
	'''
	
	steps_list = get_steps_list(preprocess_steps_and_order)

	doc_list = raw_docs

	start = time.time()
	for step in steps_list:
		if (len(preprocess_steps_and_order[step]) == 1): # The case where function does not requires arguements
			doc_list = preprocess_functions[step](doc_list, debug=debug)
		elif (step == 'make_bigrams_gensim'):
# 			print()
			doc_list = preprocess_functions[step](doc_list, preprocess_steps_and_order[step][1]['bigrams_min_count'], preprocess_steps_and_order[step][1]['bigrams_threshold'], debug)
		elif (step == 'make_trigrams_gensim'):
			doc_list = preprocess_functions[step](doc_list, preprocess_steps_and_order[step][1]['trigrams_min_count'], preprocess_steps_and_order[step][1]['trigrams_threshold'], debug)
		elif (step == 'min_max_length_removal'):
			doc_list = preprocess_functions[step](doc_list, preprocess_steps_and_order[step][1]['mmlr_min_len'], preprocess_steps_and_order[step][1]['mmlr_max_len'], preprocess_steps_and_order[step][1]['mmlr_deacc'], debug)

	
	print('~~~ pre-processing done in ', time.time()-start)
	print(' ')
	print('- Creating dictionary and corpus')
	dictionary, corpus = to_dictionary_and_corpus(doc_list, 
													fe_no_below=1, 
													fe_no_above=0.15, 
													fe_keep_n=None,
													debug=False)
	return dictionary, corpus, doc_list