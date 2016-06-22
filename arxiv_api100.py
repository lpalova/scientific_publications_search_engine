
# Download authors from arXiv

import requests
from bs4 import BeautifulSoup
import cPickle
from collections import defaultdict
import re
import string

# ### Authors, Title and Abstract from arXiv website

def process_name(full_name):
    if not full_name:
        return None
    remove = string.punctuation
    remove = remove.replace("-", "") # don't remove hyphens
    remove = remove.replace("'", "") # don't remove '
    remove = remove.replace("`", "") # don't remove `
    pattern = r"[{}]".format(remove) # create the pattern
    full_name = re.sub(pattern, "", full_name) 
    full_name = full_name.strip().split(' ')
    last_name = full_name[-1].strip()
    if not last_name:
        return None
    # e.g. Krishna Bharadwaj B to Krishna B Bharadwaj
    if (len(last_name) == 1) and (len(full_name)>1) and (len(full_name[-2].strip()) > 1):
        last_name = full_name[-2]
        full_name[-2] = full_name[-1]
    author = ''
    for name in full_name[:-1]:
        name = re.sub('-','',name)
        name = name.strip()
        if name:
            author = author + name[0].upper() + ' '
    author += last_name
    return author

def process_api(url):
	response = requests.get(url)
	if response.status_code != 200:
	    return None
	page = response.text
	soup = BeautifulSoup(page,"lxml")
	objentry = soup.find("entry")
	authors = []
	for obj in objentry.findAll("name"):
	    full_name = obj.text.encode('utf-8')
	    author = process_name(full_name)
	    if author:
	        authors.append(author)
	objtitle = objentry.find("title")
	title = objtitle.text
	objabs = objentry.find("summary")
	abstract = objabs.text
	return (authors, title, abstract)

def get_abs(text_names):
    paper_to_authors = defaultdict(list)
    set_of_authors = set()
    paper_to_title = defaultdict(str)
    paper_to_abstract = defaultdict(str)
    #paper_to_title = cPickle.load(open('./dfs/paper_to_title_fn100.p', 'rb'))
    #paper_to_abstract = cPickle.load(open('./dfs/paper_to_abstract_fn100.p', 'rb'))
    #paper_to_authors = cPickle.load(open('./dfs/paper_to_authors_fn100.p', 'rb'))
    #set_of_authors = cPickle.load(open('./dfs/set_of_authors_fn100.p', 'rb'))
    for text_name in text_names:
    	if text_name not in paper_to_title:
		    url = "http://export.arxiv.org/api/query?search_query=id:" + text_name
		    print url
		    authors, title, abstract = process_api(url)
		    paper_to_title[text_name] = title
		    paper_to_abstract[text_name] = abstract
		    paper_to_authors[text_name] = authors
		    set_of_authors.update(set(authors))
		    cPickle.dump(paper_to_title, open('./dfs/paper_to_title_fn100.p', 'wb'))
		    cPickle.dump(paper_to_abstract, open('./dfs/paper_to_abstract_fn100.p', 'wb'))
		    cPickle.dump(paper_to_authors, open('./dfs/paper_to_authors_fn100.p', 'wb'))
		    cPickle.dump(set_of_authors, open('./dfs/set_of_authors_fn100.p', 'wb')) 
    return (paper_to_authors, set_of_authors, paper_to_title, paper_to_abstract)


if __name__ == '__main__':
    text_names = cPickle.load(open('./dfs/text_names100.p', 'rb'))
    paper_to_authors, set_of_authors, paper_to_title, paper_to_abstract = get_abs(text_names)
    cPickle.dump(paper_to_title, open('./dfs/paper_to_title_fn100.p', 'wb')) 
    cPickle.dump(paper_to_abstract, open('./dfs/paper_to_abstract_fn100.p', 'wb')) 
    cPickle.dump(paper_to_authors, open('./dfs/paper_to_authors_fn100.p', 'wb')) 
    cPickle.dump(set_of_authors, open('./dfs/set_of_authors_fn100.p', 'wb')) 

