import time
import os
import gensim
from gensim.models import CoherenceModel

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