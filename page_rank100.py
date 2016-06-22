# # PageRank: authors -> refs and score for authors, papers

import cPickle
from collections import defaultdict, Counter
import numpy as np
from scipy.sparse import coo_matrix

# ### References from text

# #### authors to identifiers, identifiers to authors
def get_author_ids(set_of_authors):
    author_to_id = {}
    id_to_author = {}
    ida = 0
    for author in set_of_authors:
        author_to_id[author] = ida
        id_to_author[ida] = author
        ida += 1
    n = ida
    return (author_to_id, id_to_author, n)

# #### author to refs
def get_author_to_refs(paper_to_authors, paper_to_refs):
    author_to_refs = defaultdict(list)
    for paper in paper_to_authors:
        authors = paper_to_authors[paper]
        refs = paper_to_refs[paper]
        for author in authors:
            author_to_refs[author].extend(refs)
    return author_to_refs

def build_matrix(author_to_id, author_to_refs):
    col = []
    row = []
    weight = []
    # sum of weights for each author (column) stored as a value in dictionary col_weights with key==author(col)
    col_weights = defaultdict(int)
    for author in author_to_refs:
        author_id = author_to_id[author]
        count_refs = Counter(author_to_refs[author])
        for ref in count_refs:
            if author != ref:
                ref_id = author_to_id[ref]
                weight_value = count_refs[ref]
                col.append(author_id)
                row.append(ref_id)
                weight.append(weight_value)
                col_weights[author_id] += weight_value
    col = np.array(col)
    row = np.array(row)
    weight = np.array(weight)
    # rescale column weights
    normalized_weight = []
    for i in range(len(col)):
        normalized_weight.append(1.0*weight[i]/col_weights[col[i]])
    return (col, row, weight, normalized_weight, col_weights)

def iterate(m,d,epsilon,max_iter):
    n = m.shape[0]
    v_init = np.array([1.0/n]*n)
    v_damp = np.array([(1.0-float(d))/n]*n)
    v_old = v_init
    for i in range(max_iter):
        v_new = d*m.dot(v_old)+v_damp
        #print np.sum(v_new)
        #if np.linalg.norm(v_new-v_old,ord=np.inf) < epsilon:
        if np.linalg.norm(v_new-v_old,ord=2) < epsilon:
            break
        v_old = v_new
    else:
        return "Not converged"
    return (v_new, i)
    
# get PageRank for every author
def get_author_to_score(v, author_to_id):
    author_to_pagerank = defaultdict(float)
    for author in author_to_id:
        author_to_pagerank[author] = v[author_to_id[author]]
    return author_to_pagerank


# In[28]:

# get PageRank for every paper: mean of pageRank of authors
def get_paper_to_score(v, author_to_id, paper_to_authors):
    paper_to_pagerank = defaultdict(float)
    for paper in paper_to_authors:
        paper_score = 0.0
        n = 0
        for author in paper_to_authors[paper]:
            paper_score += v[author_to_id[author]]
            n += 1
        paper_to_pagerank[paper] = paper_score/n
    return paper_to_pagerank
    

if __name__ == '__main__':
	text_names = cPickle.load(open('./dfs/text_names100.p', 'rb'))
	paper_to_authors = cPickle.load(open('./dfs/paper_to_authors_fn100.p', 'rb'))
	set_of_authors = cPickle.load(open('./dfs/set_of_authors_fn100.p', 'rb'))
	paper_to_refs = cPickle.load(open('./dfs/paper_to_refs_fn100.p', 'rb'))

	# ### Creating a Citation Network Graph
	author_to_id, id_to_author, n_ids = get_author_ids(set_of_authors)
	cPickle.dump(author_to_id, open('./dfs/author_to_id_fn100.p', 'wb')) 
	cPickle.dump(id_to_author, open('./dfs/id_to_author_fn100.p', 'wb')) 
	#author_to_id = cPickle.load(open('./dfs/author_to_id_fn100.p', 'rb'))
	#id_to_author = cPickle.load(open('./dfs/id_to_author_fn100.p', 'rb'))
	author_to_refs = get_author_to_refs(paper_to_authors, paper_to_refs)
	cPickle.dump(author_to_refs, open('./dfs/author_to_refs_fn100.p', 'wb')) 
	#author_to_refs = cPickle.load(open('./dfs/author_to_refs_fn100.p', 'rb'))

	# #### build adjacency matrix
	col, row, weight, normalized_weight, col_weights = build_matrix(author_to_id, author_to_refs)
	m = coo_matrix((normalized_weight, (row, col)), shape=(len(set_of_authors), len(set_of_authors)))
	cPickle.dump(m, open('./dfs/m_fn100.p', 'wb')) 

	damping = 0.8
	epsilon = 0.000001
	max_iter = int(1e5)
	v, num_iter = iterate(m,damping,epsilon,max_iter)
	print 'Number of iterations: %d, Number of authors: %d' %(num_iter, len(v))
	print 'Author with max PageRank: ', id_to_author[np.argmax(v)]


	# checking that final vector of probability scores sums up to 1.0
	# v assigns score to every author

	# vyladit algorithmus: namiesto normalizacie na 1.0, skus normalizovat na pocet authors ,
	# ak pocet authors >> 1, potom v[i] << 0 a numericke errors : rounding etc.
	print 'Sum of PageRank values of all authors: ', np.sum(v)

	# #### Scoring Papers Based on Authors
	author_to_pagerank = get_author_to_score(v, author_to_id)
	paper_to_pagerank = get_paper_to_score(v, author_to_id, paper_to_authors)
	cPickle.dump(author_to_pagerank, open('./dfs/author_to_pagerank_fn100.p', 'wb')) 
	cPickle.dump(paper_to_pagerank, open('./dfs/paper_to_pagerank_fn100.p', 'wb')) 
	#author_to_pagerank = cPickle.load(open('./dfs/author_to_pagerank_fn100.p', 'rb'))
	#paper_to_pagerank = cPickle.load(open('./dfs/paper_to_pagerank_fn100.p', 'rb'))

