import string
import nltk
import sys
import networkx as nx
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

word_count = 100

sentences = []
for arg in sys.argv[1:]:
	try:
		sentences += tokenizer.tokenize(open(arg).read())
	except Exception, e:
		raise e

def normalize(sentence):
	sentence = sentence.lower()
	tokenizer = RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(sentence)
	filtered_words = [w for w in tokens if not w in stopwords.words('english')]
	return " ".join(filtered_words)

def textrank(sentences):
	matrix = CountVectorizer().fit_transform(sentences)
	normalized = TfidfTransformer().fit_transform(matrix)
	similarity_graph = normalized * normalized.T

	nx_graph = nx.from_scipy_sparse_matrix(similarity_graph)
	scores = nx.pagerank(nx_graph)

	return sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

if __name__ == '__main__':
	summary_list = zip(*textrank(sentences))[1]
	summary = ''
	for i in xrange(len(summary_list)):
		if len(summary) < word_count:
			summary += summary_list[i]
	print summary