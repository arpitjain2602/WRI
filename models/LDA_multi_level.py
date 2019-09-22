# from preprocess.preprocess import preprocess
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

def lda_model_multi_level(
				level, 
				dictionary,
				corpus,
				doc_list,
				num_topics_list_level_1,
				coherence='c_v', 
				debug=False,
				need_best_topic=True, 
				model_selection_metric='coherence',
				num_topics_list_level_2=[], 
				num_topics_list_level_3=[]
				):
	'''
	- level:
		0 - means first level, i.e. only on the documents
		1 - means 2 level, i.e. only 2 layers of documents
		2 - ...
	'''
		
	# ------------------------------------------------------------------------------------------------------------------------
	print(' ')
	print('Sample data point: ', doc_list[0])
	print(' ')

	print('### Running process for Level-1')
	lda_models_level_1 = {}
	coherence_list = []
	perplexity_list = []

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
	
	print('- Done training model for Level-1 on all topics in {} sec!'.format(time.time()-start1))
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
	best_lda_model_level_1 = lda_models_level_1[best_topic]

	lda_level_1 = {'best_lda_model': best_lda_model_level_1,
							'best_topic': best_topic,
							'coherence_score': coherence_score,
							'perplexity_score': perplexity_score,
							'corpus': corpus,
							'dictionary': dictionary,
							'doc_list': doc_list,
							'all_models': lda_models_level_1}

	



	# ------------------------------------------------------------------------------------------------------------------------
	print(' ')
	print('### Running process for Level-2')
	lda_corpus = best_lda_model_level_1[corpus]
	lda_corpus_max_topic = [max(doc[0], key=operator.itemgetter(1))[0] for doc in lda_corpus]

	corpus_cluster = [list() for i in range(best_topic)]
	for i in range(len(lda_corpus_max_topic)):
		topic = lda_corpus_max_topic[i]
		corpus_cluster[topic].append(corpus[i])

	lda_level_2 = {}
	for i in range(best_topic):
		print('~~~ Running for documents in topic', i+1)
		start = time.time()

		lda_models_level_2 = {}
		coherence_list_level_2 = []
		perplexity_list_level_2 = []
		
		for topic in num_topics_list_level_2:
			print("		Running for topic",topic)
			lda_models_level_2[topic] = gensim.models.ldamodel.LdaModel(corpus=corpus_cluster[i],
														id2word=dictionary,
														num_topics=topic,
														random_state=100,
														update_every=1,
														chunksize=100,
														passes=10,
														alpha='auto',
														per_word_topics=True)

			print('		LDA Done for {} topic! Time Taken is {}'.format(topic, time.time()-start))

			# JUST CHECK IN THE BELOW LINE IF doc_list is the actual list you need actually
			coherence, perplexity = evaluate_model(lda_models_level_2[topic], dictionary, corpus_cluster[i], doc_list, coherence = coherence)
			coherence_list_level_2.append(coherence)
			perplexity_list_level_2.append(perplexity)
			print('		Coherence - {}, Perplexity - {}'.format(coherence, perplexity))

		coherence_list_level_2 = np.array(coherence_list_level_2)
		perplexity_list_level_2 = np.array(perplexity_list_level_2)
		
		if (need_best_topic == True):
			coherence_index = np.argmax(coherence_list_level_2)
			perplexity_index = np.argmin(perplexity_list_level_2)

			if (model_selection_metric == 'coherence'):
				best_topic_index = coherence_index
			elif (model_selection_metric == 'perplexity'):
				best_topic_index = perplexity_index

		best_topic = num_topics_list_level_2[best_topic_index]
		coherence_score = coherence_list_level_2[best_topic_index]
		perplexity_score = perplexity_list_level_2[best_topic_index]
		best_lda_model = lda_models_level_2[best_topic]

		lda_level_2[i] = {'best_lda_model': best_lda_model,
							'best_topic': best_topic,
							'coherence_score': coherence_score,
							'perplexity_score': perplexity_score,
							'corpus': corpus_cluster[i],
							'dictionary': dictionary,
							'doc_list': doc_list,
							'all_models': lda_models_level_2}

		print('~~~ Process done for documents in topic', i+1)

	return lda_level_1, lda_level_2



