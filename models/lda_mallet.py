# from utils.utils import get_topics, evaluate_model
import pickle
import time
import numpy as np
import operator
import pandas as pd
import gensim
from gensim.models import CoherenceModel
import time
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import pickle

def get_topics(lda_model):
	topics = lda_model.print_topics()
	print("~~~ Topics are:")
	for i in range(len(topics)):
		print('Topic ',i)
		print(topics[i][1])
		print(' ')

def evaluate_model(lda_model, dictionary, corpus, doc_list, coherence = 'c_v', debug=False):
	'''
	C_v measure is based on a sliding window, one-set segmentation of the top words and an indirect confirmation measure that uses normalized pointwise mutual information (NPMI) and the cosine similarity
	C_p is based on a sliding window, one-preceding segmentation of the top words and the confirmation measure of Fitelson’s coherence
	C_uci measure is based on a sliding window and the pointwise mutual information (PMI) of all word pairs of the given top words
	C_umass is based on document cooccurrence counts, a one-preceding segmentation and a logarithmic conditional probability as confirmation measure
	C_npmi is an enhanced version of the C_uci coherence using the normalized pointwise mutual information (NPMI)
	C_a is baseed on a context window, a pairwise comparison of the top words and an indirect confirmation measure that uses normalized pointwise mutual information (NPMI) and the cosine similarity
	'''
	start = time.time()

	# Compute Perplexity
	perplexity = lda_model.log_perplexity(corpus)
	if(debug):
		print('Perplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

	# Compute Coherence Score
	coherence_model_lda = CoherenceModel(model=lda_model, texts=doc_list, dictionary=dictionary, coherence='c_v')
	coherence_lda = coherence_model_lda.get_coherence()
	if(debug):
		print('Coherence Score: ', coherence_lda)

	end = time.time()
	if(debug):
		print('\nTime Taken: ', end-start)
	return coherence_lda, perplexity

def lda_mallet(
				dictionary,
				corpus,
				doc_list,
				num_topics_list_level_1,
				coherence='c_v', 
				debug = False,
				need_best_topic=True, 
				model_selection_metric='coherence'):
	'''
	- runs LDA at 1 level
	'''
	
	lda_models_level_1 = {}
	coherence_list = []
	perplexity_list = []
	print(' ')
	print('Sample data point: ', doc_list[0])
	print(' ')
	
	start1 = time.time()

	for topic in num_topics_list_level_1:
		print('	### Running LDA for number of topic - {}'.format(topic))
		start = time.time()
		
		mallet_path = '/Users/arpitjain/Desktop/WRI/models/mallet-2.0.8/bin/mallet.bat' # update this path

		lda_models_level_1[topic] = gensim.models.wrappers.LdaMallet(mallet_path,
														corpus=corpus,
														id2word=dictionary,
														num_topics=topic)

		print('	Mallet LDA Done for {} topic! Time Taken is {}'.format(topic, time.time()-start))

		print('	Evaluating model for number of topic - {}'.format(topic))

		coherence, perplexity = evaluate_model(lda_models_level_1[topic], dictionary, corpus, doc_list, coherence = coherence)
		coherence_list.append(coherence)
		perplexity_list.append(perplexity)
		print('Coherence - {}, Perplexity - {}'.format(coherence, perplexity))
		print('---')
	
	print('- Done training model on all topics in {} sec!'.format(time.time()-start1))
	coherence_list = np.array(coherence_list)
	perplexity_list = np.array(perplexity_list)
	
	if (need_best_topic == True):
		coherence_index = np.argmax(coherence_list)
		perplexity_index = np.argmin(perplexity_list)

		if (model_selection_metric == 'coherence'):
			best_topic_index = coherence_index
		elif (model_selection_metric == 'perplexity'):
			best_topic_index = perplexity_index

	best_topic = num_topics_list_level_1[best_topic_index]
	coherence_score = coherence_list[best_topic_index]
	perplexity_score = perplexity_list[best_topic_index]
	best_lda_model = lda_models_level_1[best_topic]

	lda_level_1 = {'best_lda_model': best_lda_model,
							'best_topic': best_topic,
							'coherence_score': coherence_score,
							'perplexity_score': perplexity_score,
							'corpus': corpus,
							'dictionary': dictionary,
							'doc_list': doc_list,
							'all_models': lda_models_level_1,
							'coherence_score_list': coherence_list,
							'perplexity_score_list': perplexity_list}

	print('Done Single-Level LDA')
	return lda_level_1