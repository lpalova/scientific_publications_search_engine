# # LSI: paper by content

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re
import string
import matplotlib.pyplot as plt
import cPickle
#import nltk
from textblob import TextBlob
from nltk.stem.snowball import SnowballStemmer
# gensim, sklearn
from gensim import models, matutils
from sklearn.utils.extmath import randomized_svd
import matplotlib
matplotlib.style.use('ggplot')

# ### Text analysis

def clean_text(text):
    text = text.lower()
    printable = set(string.printable)
    text = filter(lambda x: x in printable, text)
    stemmer = SnowballStemmer('english', ignore_stopwords=True)
    clean_list = []
    blob = TextBlob(text)
    for word in blob.words:
        clean_list.append(stemmer.stem(word))
    text = ' '.join(clean_list)
    text = re.sub('\r',' ',text) #all new lines
    text = re.sub('\d',' ',text) #all numbers
    text = re.sub('\n',' ',text) #all new lines
    text = re.sub('['+string.punctuation+']',' ',text) #all punctuation
    text = re.sub('http\S+','',text) #all urls 
    text = re.sub(r'\W*\b\w{1,2}\b','',text) # remove 1-2 character words
    return text


# #### Stop words
def create_stop_list(path):
    stop_df = pd.read_csv(path, header =None, names=['word'])
    stop_list = list(stop_df.word)
    stop_list += ["et","al","fig","figure","image","histogram","plot", "table",
             "use","shown","result","seen","see","known","som","learn","fix",
             "ct","ev","eq","eqs","equation","data","ndata", "nm","cm","mm","ii","iii","iv","vi","vii","viii","ix","vs",
             "unit","units","term",
             "science", "electric", "magnetic", "field","energy","work",
             "resp","respectively",
             "url","http","arxiv","doi","references","abstract","introduction",
             "sec","section","conclusion","discussion",
             "thz","hz","mhz","nhz","khz","gpa","tpa","mpa","kpa",
             "solid", "state", "comm","journal","phys","rev","lett","appl",
             "physrevb","physreva","review","link","aps","org",
             "nature","chemistry","principles","springer-verlag","springer", "verlag","sov","jetp", "alpha","beta","gamma","delta","epsilon","zeta","eta","theta","iota","kappa","lambda","mu",
              "nu","xi","omicron","pi","rho","sigma","tau","upsilon", "phi","chi","psi","omega"]
    stop_list = clean_text(" ".join(stop_list))
    stop_list = TextBlob(stop_list).words
    return stop_list


# LSI on Sample of 100+ papers
if __name__ == '__main__':
	path = "./txts/"
	text_names = cPickle.load(open('./dfs/text_names100.p', 'rb'))
	text_list = []
	for name in text_names:
		with open(path + name + ".txt", 'r') as f:
		    text = f.read()
		text = clean_text(text)
		text_list.append(text)  
	cPickle.dump(text_list, open('./dfs/text_list100.p', 'wb')) 
	#text_list = cPickle.load(open('./dfs/text_list100.p', 'rb'))

	# ### CountVectorizer
	stop_list = create_stop_list('./dfs/fox_1990_stoplist.txt')
	count_vectorizer = CountVectorizer(analyzer='word',
		                              ngram_range=(1, 2), stop_words = stop_list,#stop_words='english',
		                              token_pattern='\\b[a-z][a-z]+\\b'#,#min_df=2, max_df=0.9
		                              )
	count_vectorizer.fit(text_list)
	cPickle.dump(count_vectorizer, open('./dfs/count_vectorizer100.p', 'wb')) 
	#count_vectorizer = cPickle.load(open('./dfs/count_vectorizer100.p', 'rb'))

	#X_SVD = count_vectorizer.transform(text_list)
	#U, Sigma, VT = randomized_svd(X_SVD, n_components=100, n_iter=5, random_state=None)
	#Sigma_norm = Sigma / np.linalg.norm(Sigma)
	#plt.figure(figsize=(10,6.5))
	#plt.scatter(range(len(Sigma_norm)),Sigma_norm, s =40, color = 'darkblue')
	#plt.xlabel("Index",fontsize=30)
	#plt.ylabel("Eigenvalue",fontsize=30)
	#plt.xticks(fontsize=20)
	#plt.yticks(fontsize=20)
	#plt.xlim(-5,100,1)
	#plt.savefig('./figs/SVD_sigma100.png')
	##plt.show()

    # Create a Count_vectorizer Corpus, and id2word map
    # Convert sparse matrix (from count_vectorizer) of counts to a gensim corpus
	text_vecs = count_vectorizer.transform(text_list).transpose()
	corpus = matutils.Sparse2Corpus(text_vecs)
	cPickle.dump(corpus, open('./dfs/corpus100.p', 'wb')) 
	id2word = dict((v, k) for k, v in count_vectorizer.vocabulary_.iteritems())
	cPickle.dump(id2word, open('./dfs/id2word100.p', 'wb')) 

	# Create a TFIDF transformer from our word counts (equivalent to "fit" in sklearn)
	tfidf = models.TfidfModel(corpus)
	cPickle.dump(tfidf, open('./dfs/tfidf100.p', 'wb')) 
	#corpus = cPickle.load(open('./dfs/corpus100.p', 'rb'))
	#tfidf = cPickle.load(open('./dfs/tfidf100.p', 'rb'))
	#id2word = cPickle.load(open('./dfs/id2word100.p', 'rb'))
	# Create a TFIDF vector for all documents from the original corpus ("transform" in sklearn)
	tfidf_corpus = tfidf[corpus]
	doc_tfidf = [doc for doc in tfidf_corpus]
	cPickle.dump(doc_tfidf, open('./dfs/doc_tfidf100.p', 'wb'))
	
	# Build an LSI space from the input TFIDF matrix, mapping of row id to word, and num_topics
	# num_topics is the number of dimensions to reduce to after the SVD
	# Analagous to "fit" in sklearn, it primes an LSI space
	lsi = models.LsiModel(tfidf_corpus, id2word=id2word, num_topics=90)
	#cPickle.dump(lsi, open('./dfs/lsi100.p', 'wb')) 
	lsi.save('./dfs/lsi100.save')

	# Retrieve vectors for the original tfidf corpus in the LSI space ("transform" in sklearn)
	# Dump the resulting document vectors into a list
	lsi_corpus = lsi[tfidf_corpus]
	doc_vecs = [doc for doc in lsi_corpus]
	cPickle.dump(doc_vecs, open('./dfs/doc_vecs100.p', 'wb')) 

	# ### Clustering with LSI Vectors
	# Convert the gensim-style corpus vecs to a numpy array for sklearn manipulations
	X = matutils.corpus2dense(lsi_corpus, num_terms=90).transpose()
	cPickle.dump(X, open('./dfs/X_lsi_corpus100.p', 'wb')) 

