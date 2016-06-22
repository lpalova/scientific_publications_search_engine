# # m to input for gephi

import cPickle
from collections import defaultdict, Counter
import csv
from sklearn.cluster import DBSCAN, AgglomerativeClustering

# get topic clustering for papers
def get_topics(X_lsi, text_names, nk=1):
    ag = AgglomerativeClustering(n_clusters=nk, affinity='cosine', linkage='average')
    topics = ag.fit_predict(X_lsi)
    paper_to_topic = defaultdict(int)
    topic_to_papers = defaultdict(list)
    for paper,topic in zip(text_names,topics):
        paper_to_topic[paper] = topic
        topic_to_papers[topic].append(paper)
    return (paper_to_topic, topic_to_papers)

def get_topic_words(topic_to_papers, id2word, text_names, n):
    topic_to_words = defaultdict(str)
    for topic in topic_to_papers:
       word_to_weight = defaultdict(float)
       papers = topic_to_papers[topic]
       for paper in papers:
           topnwords = doc_tfidf[text_names.index(paper)]
           for tup in topnwords:
               word_to_weight[id2word[tup[0]]] += tup[1]
       for key, value in Counter(word_to_weight).most_common(n):
           topic_to_words[topic] += key + ','
    return topic_to_words

def get_author_to_topic(paper_to_authors, paper_to_topic):
    author_to_topic = defaultdict(int)
    author_to_topics = defaultdict(list)
    for paper in paper_to_authors:
        for author in paper_to_authors[paper]:
            author_to_topics[author].append(paper_to_topic[paper])
    for author in author_to_topics:
        topic = Counter(author_to_topics[author]).most_common(1)[0][0]
        author_to_topic[author] = topic
    return author_to_topic

def get_refs_set(paper_to_refs):
    refs_set = set()
    for paper in paper_to_refs:
        refs_set.update(set(paper_to_refs[paper]))
    return refs_set
 
def get_nodes_edges(m, refs_set, id_to_author, author_to_topic, author_to_pagerank):
    nodes_set = set()
    list_of_nodes = [['Id','Label','Topic','PageRankScore']]
    list_of_edges = [['Source','Target','Weight']]
    for i,j,w in zip(m.row, m.col, m.data):
        if id_to_author[i] in refs_set:
            nodes_set.add(i)
            nodes_set.add(j)
            list_of_edges.append([i,j,w])
        else:
            print id_to_author[i], id_to_author[j], w
    for node in nodes_set:
        author = id_to_author[node]
        topic = author_to_topic[author]
        pagerank = author_to_pagerank[author]
        list_of_nodes.append([node,author,topic,pagerank])
    return (list_of_nodes, list_of_edges)

if __name__ == '__main__':	
    m = cPickle.load(open('./dfs/m_fn100.p', 'rb'))
    id_to_author = cPickle.load(open('./dfs/id_to_author_fn100.p', 'rb'))
    paper_to_authors = cPickle.load(open('./dfs/paper_to_authors_fn100.p', 'rb'))
    paper_to_refs = cPickle.load(open('./dfs/paper_to_refs_fn100.p', 'rb'))
    text_names = cPickle.load(open('./dfs/text_names100.p', 'rb'))
    doc_tfidf = cPickle.load(open('./dfs/doc_tfidf100.p', 'rb'))
    id2word = cPickle.load(open('./dfs/id2word100.p', 'rb'))
    author_to_pagerank = cPickle.load(open('./dfs/author_to_pagerank_fn100.p', 'rb'))
    paper_to_pagerank = cPickle.load(open('./dfs/paper_to_pagerank_fn100.p', 'rb'))
    X_lsi = cPickle.load(open('./dfs/X_lsi_corpus100.p', 'rb'))
    N_words = 10
    N_corpus = 100
    
    # Topic clustering    
    # Run clustering algorithm for ALL papers in corpus
    N_clusters_set = set()
    for N_percluster in range(1,N_corpus):
        N_clusters_set.add(int(N_corpus/N_percluster))
    
    for N_clusters in N_clusters_set:
        # Get topic for paper
        print N_clusters
        paper_to_topic, topic_to_papers = get_topics(X_lsi, text_names, N_clusters)
        cPickle.dump(paper_to_topic, 
            open('./dfs/paper_to_topic_fn100_'+ str(N_clusters) + '.p', 'wb')) 
        cPickle.dump(topic_to_papers, 
            open('./dfs/topic_to_papers_fn100_' + str(N_clusters) + '.p', 'wb')) 
        # Get topic words
        topic_to_words = get_topic_words(topic_to_papers, id2word, text_names, N_words)
        cPickle.dump(topic_to_words,
            open('./dfs/topic_to_words_fn100_' + str(N_clusters) + '.p', 'wb'))
    
    # Special case: N_clusters = 8    
    N_clusters = 10
    print N_clusters
    paper_to_topic, topic_to_papers = get_topics(X_lsi, text_names, N_clusters)
    cPickle.dump(paper_to_topic, 
        open('./dfs/paper_to_topic_fn100_'+ str(N_clusters) + '.p', 'wb')) 
    cPickle.dump(topic_to_papers, 
        open('./dfs/topic_to_papers_fn100_' + str(N_clusters) + '.p', 'wb')) 
    # Get topic words
    topic_to_words = get_topic_words(topic_to_papers, id2word, text_names, N_words)
    cPickle.dump(topic_to_words,
        open('./dfs/topic_to_words_fn100_' + str(N_clusters) + '.p', 'wb'))
    
    # Get topic for author    
    author_to_topic = get_author_to_topic(paper_to_authors, paper_to_topic)
    cPickle.dump(author_to_topic, 
        open('./dfs/author_to_topic_fn100_' + str(N_clusters) + '.p', 'wb'))
    
    # Get set of references: those authors who are not in the refs set
    # will be excluded from the citation network, nobody points to them
    refs_set = get_refs_set(paper_to_refs)
    cPickle.dump(refs_set, open('./dfs/set_of_refs100.p', 'wb'))
    
    # Get nodes and edges for graph
    list_of_nodes, list_of_edges = get_nodes_edges(m, refs_set, id_to_author, author_to_topic, author_to_pagerank)
    
    with open("./gephi/nodes100.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(list_of_nodes)
    with open("./gephi/edges100.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(list_of_edges)
        
    
    

