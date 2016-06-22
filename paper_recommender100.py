from flask import Flask
from flask import request, redirect
from flask import render_template
import cPickle
import pandas as pd
import re
from nltk.stem.snowball import SnowballStemmer
from textblob import TextBlob
import string
from gensim import similarities, matutils, models
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering

#---------- MODEL IN MEMORY ----------------#

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

def create_stop_list(path):
    stop_df = pd.read_csv(path, header =None, names=['word'])
    stop_list = list(stop_df.word)
    stop_list += ["et","al","fig","figure","phys","rev","lett","appl","table",
             "use","shown","result","seen","see","known",
             "ct","ev","eq","eqs","equation", "nm","cm","mm","ii","iii","iv","vi","vii","viii","ix","vs",
             "unit","units","sec","section","term",
             "science", "electric", "magnetic", "field","energy","work",
             "resp","respectively",
             "url","http","arxiv",       "alpha","beta","gamma","delta","epsilon","zeta","eta","theta","iota","kappa","lambda","mu",
              "nu","xi","omicron","pi","rho","sigma","tau","upsilon", "phi","chi","psi","omega"]
    stop_list = clean_text(" ".join(stop_list))
    stop_list = TextBlob(stop_list).words
    return stop_list

def process_text_blob(text_blob):
    stop_list = create_stop_list('fox_1990_stoplist.txt')
    text_blob = clean_text(text_blob)
    text_blob = TextBlob(text_blob)
    text_blob = [word for word in text_blob.words if word not in stop_list]
    text_blob = " ".join(text_blob)
    return text_blob

# Return list of indexes of documents with maximum cosine similarities to document k
def most_similar_bycontent(text_names, doc_vecs, n=20, paper ='', k=0):
    shift = 0
    if paper:
        k = text_names.index(paper)
    else:
        shift = 1
    index_list = similarities.MatrixSimilarity(doc_vecs)[doc_vecs[k]]
    l = sorted(enumerate(index_list), key=lambda x: x[1], reverse=True )[1:n+1]
    most_similar_indices = [item[0]-shift for item in l]
    text_names_by_content = []
    for i in most_similar_indices:
        text_names_by_content.append(text_names[i])
    return text_names_by_content

# get NORMALIZED cosine_similarity for every paper (to the queried paper)
def norm_cosine_similarity(text_names, doc_vecs, paper):
    paper_to_cossim = defaultdict(float)
    k = text_names.index(paper)
    sim_scores = similarities.MatrixSimilarity(doc_vecs)[doc_vecs[k]]
    #norm = sum(sim_scores)-1.0
    norm = sum(sim_scores)
    for a,b in zip(text_names, sim_scores):
        #if a != paper:
        #    paper_to_cossim[a] = b/norm
        paper_to_cossim[a] = b/norm
    return paper_to_cossim

# get NORMALIZED PageRank for every paper
def norm_pagerank(paper_to_pagerank, paper):
    norm = 0.0
    paper_to_norm_pagerank = defaultdict(float)
    for item in paper_to_pagerank:
        #if item != paper:
        #    norm += paper_to_pagerank[item]
        norm += paper_to_pagerank[item]
    for item in paper_to_pagerank:
        paper_to_norm_pagerank[item] = paper_to_pagerank[item]/norm
    #paper_to_norm_pagerank.pop(paper, None)
    return paper_to_norm_pagerank

# Input: paper ID, list of paper IDs, list of document vectors, dictionary of (paper,PageRank_score) pairs,
# tunning parameter alpha: alpha==0 for content-based search, alpha==1 for citations-based search
# Output: dictionary of (paper,score) pairs, where score is a combination of content and citations PageRank
def get_paper_score(paper, text_names, doc_vecs, paper_to_pagerank,alpha=0.0):
    paper_to_norm_cossim = norm_cosine_similarity(text_names, doc_vecs, paper)
    paper_to_norm_pagerank = norm_pagerank(paper_to_pagerank, paper)
    #print paper_to_norm_cossim
    #print paper_to_norm_pagerank
    paper_to_score = defaultdict(float)
    #na = 0.0
    #nb = 0.0
    for item in paper_to_norm_cossim:
        #na += paper_to_norm_cossim[item]
        #nb += paper_to_norm_pagerank[item]
        paper_to_score[item] = ((1-alpha)*paper_to_norm_cossim[item]) + (alpha*paper_to_norm_pagerank[item])
    #print na, nb
    return paper_to_score

# get subset of papers_to_score for papers that share an author or reference with 'paper'
def get_subset_papers(paper, paper_to_authors, paper_to_refs, paper_to_score, filterby,paper_to_topic):
    subset_paper_to_score = defaultdict(float)
    if filterby == 2: # filter by authors and references only
        authorsandrefs = set(paper_to_authors[paper]).union(set(paper_to_refs[paper]))
        for item in paper_to_score:
            #persons = (set(paper_to_authors[item])).union(set(paper_to_refs[item]))
            persons = set(paper_to_authors[item])
            if persons.intersection(authorsandrefs):
                subset_paper_to_score[item] = paper_to_score[item]
    else: #filter by topics
        my_topic = paper_to_topic[paper]
        for item in paper_to_score:
            if paper_to_topic[item] == my_topic:
                subset_paper_to_score[item] = paper_to_score[item]
    return subset_paper_to_score       

# get ranked subset of papers based on score
def get_ranked_subset(text_names, doc_vecs, paper_to_pagerank, paper_to_authors, paper_to_refs, paper, n, alpha,filterby,paper_to_topic):
    paper_to_score = get_paper_score(paper,text_names,doc_vecs,paper_to_pagerank,alpha)
    if filterby != 1:
        subset_paper_to_score = get_subset_papers(paper, paper_to_authors, paper_to_refs, paper_to_score,filterby,paper_to_topic)
    else:
        subset_paper_to_score = paper_to_score
    return [a for (a,b) in sorted(subset_paper_to_score.iteritems(),key=lambda (k,v): v,reverse=True)][:n]

def get_options(my_list):
    s = []
    for item in my_list:
        s.append('<option value="' + item + '">' + item + '</option>')
    return s

N_corpus = 100
#Topic_Step = max(int(0.01*(N_corpus)),1)
N_words = 10
text_names = cPickle.load(open('text_names100.p', 'rb'))
count_vectorizer = cPickle.load(open('count_vectorizer100.p', 'rb'))
tfidf = cPickle.load(open('tfidf100.p', 'rb'))
lsi = models.LsiModel.load('lsi100.save')
doc_vecs = cPickle.load(open('doc_vecs100.p', 'rb'))
doc_tfidf = cPickle.load(open('doc_tfidf100.p', 'rb'))
id2word = cPickle.load(open('id2word100.p', 'rb'))
author_to_pagerank = cPickle.load(open('author_to_pagerank_fn100.p', 'rb'))
paper_to_pagerank = cPickle.load(open('paper_to_pagerank_fn100.p', 'rb'))
paper_to_authors = cPickle.load(open('paper_to_authors_fn100.p', 'rb'))
paper_to_refs = cPickle.load(open('paper_to_refs_fn100.p', 'rb'))
paper_to_title = cPickle.load(open('paper_to_title_fn100.p', 'rb'))
paper_to_abstract = cPickle.load(open('paper_to_abstract_fn100.p', 'rb'))
author_to_refs = cPickle.load(open('author_to_refs_fn100.p', 'rb'))
X_lsi = cPickle.load(open('X_lsi_corpus100.p', 'rb'))
options = "\n".join(get_options(text_names))

#---------- STRINGS            -------------#

def get_list_of_papers(my_list, quantity, paperid, alpha, filterby, clsize,
                       paper_to_title, paper_to_authors, paper_to_abstract,
                       paper_to_topic,topic_to_words):
    s = []
    for item in my_list:
        authors_string = ""
        for i in range(len(paper_to_authors[item])):
            author = paper_to_authors[item][i]
            authors_string += str(author) 
            if i < len(paper_to_authors[item])-1:
                authors_string += ", " 
        path = "http://arxiv.org/pdf/" + item + ".pdf"
        if item != paperid:
            s.append("<li><div id=\"searchresults\">")
        else:
            s.append("<li><div id=\"searchpaper\">")
        s.append("<a href='" + path + "'>" + 
                  paper_to_title[item] + "</a><br>" + 
                  '<i>Topic ' + str(paper_to_topic[item]) + ': ' +
                  topic_to_words[paper_to_topic[item]] + '...' +
                  "</i><br><small><i><font color=\"grey\">" + 
                  authors_string.decode("utf-8") + 
                  "</font></i><br>" +
                  paper_to_abstract[item] +
                 "</small><a href='/recommend?quantity=" + 
                 str(quantity) + 
                 "&paperid=" + item + 
                 "&alpha=" + str(alpha) +
                 "&filterby=" + str(filterby) +
                 "&clsize=" + str(clsize) +
                 "'><font color=\"green\">similar</font></a>" +
                 "<br></font></li></div>")
    return s

def get_form_string(text,quantity,paperid,alpha,filterby,clsize,options,
paper_to_title, paper_to_authors, paper_to_abstract, paper_to_topic, topic_to_words):
    if filterby == 2:
        threechoices = '''<option value="2">Authors and References Only</option>
                        <option value="3">Topic Only</option>
                        <option value="1">All Papers</option>'''
    elif filterby == 3:
        threechoices = '''<option value="3">Topic Only</option>
                        <option value="1">All Papers</option>
                        <option value="2">Authors and References Only</option>'''
    else:
        threechoices = '''<option value="1">All Papers</option>
                        <option value="2">Authors and References Only</option>
                        <option value="3">Topic Only</option>
                        '''
    paperid_box = ""
    if paperid:
        paperid_box = "<div id=\"searchby\">"
        authors_string = ""
        for i in range(min(6,len(paper_to_authors[paperid]))):
            author = paper_to_authors[paperid][i]
            authors_string += str(author) 
            if i < len(paper_to_authors[paperid])-1:
                authors_string += ", " 
        if i < len(paper_to_authors[paperid])-1:
            authors_string += '...'
        path = "http://arxiv.org/pdf/" + paperid + ".pdf"
    else:
        paperid_box = "<div id=\"searchnotby\">"
    paperid_box += "<label><strong>Search By Paper ID:</strong></label><br>"
    paperid_box += "<input type=\"text\" list=\"ids\" name=\"paperid\""
    paperid_box += "style=\"width: 200px;  height: 20px;\""
    paperid_box += "value=\"" + paperid + "\"/input>"
    paperid_box += "<datalist id=\"ids\">" + options + "</datalist><br>"
    if paperid:
        paperid_box += "<a href='" + path + "'>"
        paperid_box += paper_to_title[paperid] + "</a>"
        paperid_box += "<br><small><i><font color=\"grey\">"
        paperid_box += authors_string.decode("utf-8") + "</font></i></small><br>"
        paperid_box += '<i> Topic ' + str(paper_to_topic[paperid]) + ': '
        paperid_box += topic_to_words[paper_to_topic[paperid]] + '...</i><br>'
    #paperid_box += '</div>' 
    text_box = ""
    if text:
        text_box = "<div id=\"searchby\">"
    text_box += "<label><strong>Search By Text Query </strong>"
    text_box += "<small>(Reset Paper ID)</small>:</label><br>"
    text_box += "<input type=\"text\" name=\"query\""
    text_box += "style=\"width: 1000px;  height: 20px;\"value=\""  + text 
    text_box += "\"></input><br>"
    if text:
        text_box += "</div>" 
    return ('''
		<form action="/recommend" method="get">''' +
		  text_box + '''<br>''' +
		  paperid_box + 
		  '''<br><label><strong>Content (0.0) to Citation (1.0) mix: ''' 
		  + str(alpha) + '''</strong></label>
		  <br>
          <input type="range" name="alpha" id="alpha" min="0.0" max="1.0" step="0.01"
          value="''' +
          str(alpha) + '''"></input>
          <br>
          <label><strong>Filter By:</strong></label>
          <select name="filterby" style="width: 200px; height: 30px; color:navy">
          ''' + threechoices + '''
          </select>
          </div>
          <br>
          <label><strong>Papers Per Cluster:</strong></label>
          <input type="number" id="clsize" name="clsize" min="1" 
          max="''' + str(N_corpus) + '''"
          step="''' + str(1) + '''"
          style="width: 50px; height: 20px;" value="''' +
          str(clsize) + '''"></input>
          <br>
		  <label><strong>Search Results </strong><small>(max.500)</small>:</label>
          <input type="number" name="quantity" min="1" max="''' +
          str(min(500,N_corpus)) + '''" 
          style="width: 50px; height: 20px;" value="''' +
          str(quantity) + '''"></input>
          <br>
		  <input type="submit" id="go" value="Go!"></input>
        </form>
	''')
#(between 1 and quantity)
#(between 1 and min(corpus,500))
#'''<br>''' +  paper_to_abstract[paperid][:400] + '...' + 
#---------- URLS AND WEB PAGES -------------#

# Initialize the app
app = Flask(__name__)
app.config['PROPAGATE_EXCEPTIONS'] = True

# Homepage
@app.route("/")
def viz_page():
    with open("recommender.html", 'r') as viz_file:
        return viz_file.read()

@app.route('/recommend', methods = ['GET'])
def recommend():
    text = request.args.get('query')
    paperid = request.args.get('paperid')
    quantity = request.args.get('quantity')
    alpha = request.args.get('alpha')
    filterby = request.args.get('filterby')
    clsize = request.args.get('clsize')
    if quantity:
        quantity = int(quantity)
    else:
        quantity = 10
    if alpha:
        alpha = float(alpha)
    else:
        alpha = 0.0
    if filterby:
        filterby = int(filterby)
    else:
        filterby = 1
    if clsize:
        clsize = int(clsize)
        if clsize > N_corpus:
            clsize = N_corpus
    else:
        clsize = 1
    N_clusters = int(N_corpus/clsize)
    paper_to_topic = cPickle.load(open('paper_to_topic_fn100_' + str(N_clusters) + '.p', 'rb')) 
    topic_to_papers = cPickle.load(open('topic_to_papers_fn100_' + str(N_clusters) + '.p', 'rb')) 
    topic_to_words = cPickle.load(open('topic_to_words_fn100_' + str(N_clusters) + '.p', 'rb')) 
    if not text: text = ""
    if not paperid: paperid = ""
    results_list = []
    if text:
        text_blob = process_text_blob(text)
        test_vecs = count_vectorizer.transform([text_blob]).transpose()
        test_corpus = matutils.Sparse2Corpus(test_vecs)
        test_tfidf = tfidf[test_corpus]
        test_lsi = lsi[test_tfidf]
        doc_test_vecs = [doc for doc in test_lsi]
        full_docs = doc_test_vecs + doc_vecs
        results_list = list(most_similar_bycontent(text_names, full_docs, quantity))
    if paperid:
        results_list = list(get_ranked_subset(text_names, doc_vecs, paper_to_pagerank, paper_to_authors, paper_to_refs, paperid, quantity, alpha,filterby,paper_to_topic))
    if text and paperid:
        text = ""
    #cPickle.dump(results_list, open('../dapp/searchresults_list100.p', 'wb'))
    form_string = get_form_string(text, quantity, paperid, alpha,filterby,clsize,options,paper_to_title, paper_to_authors,
    paper_to_abstract,paper_to_topic,topic_to_words)  
    out = '''
    <!doctype html>
    <html>
    <head>
    <style>
     form {
        overflow: hidden;
        font-family:"Comic Sans MS";
        font-size: 15pt;
        padding: 20px;
     }
     H1 {
        font-size: 20pt;
        font-family:"Comic Sans MS";
        height: 20px;
        padding: 15px;
        margin: 2px;
     }
     ul {
        font-size: 15pt;
     }
     #searchby {
        background-color: #FAEBD7;
        width: 1000px;
        padding: 15px;
        border: 5px solid navy;
        margin: 2px;
        font-family:"Comic Sans MS";
    }  
    #searchpaper {
        background-color: #FAEBD7;
        width: 1000px;
        padding: 15px;
        margin: 2px;
        font-family:"Comic Sans MS";
    } 
    #searchnotby {
        background-color: #E5E4E2;
        width: 1000px;
        padding: 15px;
        border: 5px solid navy;
        margin: 2px;
        font-family:"Comic Sans MS";
    } 
    #searchresults {
        background-color: #FEFCFF;
        width: 1000px;
        padding: 15px;
        margin: 2px;
        font-family:"Comic Sans MS";
    }  
    input[type='range']{
        -webkit-appearance: none;
         background:#C0C0C0;
         cursor: pointer;
         display: block;
         height: 15px;
         width: 330px; 
         padding: 2px;
    }
    input[type='range']::-webkit-slider-thumb {
       -webkit-appearance: none;
        background:#191970;
        height:25px;
        width:10px;
    }
    </style>
    </head>
    <body>
    %s
    <H1><i>Search Results</i></H1>
    <ul>
    %s
    </ul>
    </body></html>
    ''' % (form_string,
    "\n".join(get_list_of_papers(results_list,quantity,paperid,alpha,filterby,clsize,paper_to_title,
    paper_to_authors,paper_to_abstract,paper_to_topic,topic_to_words)))
    return out

##6495ED
#--------- RUN WEB APP SERVER ------------#
if __name__ == '__main__':
	# Start the app server on port 80
	# (The default website port)
	#app.run(host='0.0.0.0', port=80)
	app.run(host='localhost', port=8080, debug=True)
    #app.run(host='0.0.0.0')
    #app.run()

