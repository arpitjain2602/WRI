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


def lda_model_single_level(
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
		
		lda_models_level_1[topic] = gensim.models.ldamodel.LdaModel(corpus=corpus,
														id2word=dictionary,
														num_topics=topic,
														random_state=100,
														update_every=1,
														chunksize=100,
														passes=10,
														alpha='auto',
														per_word_topics=True)

		print('	LDA Done for {} topic! Time Taken is {}'.format(topic, time.time()-start))

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
							'all_models': lda_models_level_1}

	print('Done Single-Level LDA')
	return lda_level_1