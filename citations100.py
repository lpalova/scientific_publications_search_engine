
# # text -> get references

import re
import cPickle
from collections import defaultdict

# ### References from text

def process_names(set_of_authors):
    set_of_lastnames = set()
    lastname_to_authors = defaultdict(list)
    for author in set_of_authors:
        full_name = author.split(' ')
        last_name = full_name[-1]
        set_of_lastnames.add(last_name)
        lastname_to_authors[last_name].append(full_name)
    return (set_of_lastnames, lastname_to_authors)

def authors_in_text(text, set_of_authors):
    author_occ = []
    set_of_lastnames, lastname_to_authors = process_names(set_of_authors)
    n_text = len(text)
    for i in range(n_text):
        if text[i] in set_of_lastnames:
            authors = lastname_to_authors[text[i]]
            authors = sorted(authors, key=lambda x:len(x), reverse = True)
            for author in authors:
                num_initials = len(author)-1
                tl = text[i-num_initials:i]
                tr = text[i+1:i+num_initials+1]
                l_counter = 0
                r_counter = 0
                go_tl = (len(tl)==num_initials)
                go_tr = (len(tr)==num_initials)
                for j in range(num_initials):
                    tl[j] = re.sub('-','',tl[j])
                    tr[j] = re.sub('-','',tr[j])
                    if go_tl and (tl[j]) and (tl[j][0].upper() == author[j]):
                        l_counter += 1
                    if go_tr and (tr[j]) and (tr[j][0].upper() == author[j]):
                        r_counter += 1
                if (l_counter == num_initials) or (r_counter == num_initials):
                    author_occ.append(" ".join(author))
                    break
                # OLD CODE
                #first_initial = ''
                #if len(author) > 1:
                #    first_initial = author[0]
                #if (text[i-num_initials:i] == author[:-1]) | (text[i+1:i+num_initials+1] == author[:-1]):
                #    author_occ.append(" ".join(author))
                #searches only for first_initial
                #elif (text[i-1]) and (text[i-1][0] == first_initial):
                #    author_occ.append(" ".join(author))
                #elif (text[i+1]) and (text[i+1][0] == first_initial):
                #    author_occ.append(" ".join(author))
                #else:
                #    for j in range(num_initials):
                #        if (i-num_initials+j < 0):
                #            break
                #        if (i+j+1) >= n_text:
                #            break
                #        if (text[i-num_initials+j][0].upper() != author[j]) and 
                #        (text[i+j+1][0].upper() != author[j]):
                #            break
                #    else:
                #        
                #        author_occ.append(" ".join(author))
    return author_occ

def get_refs(text_names, set_of_authors):
    paper_to_refs = defaultdict(list)
    main_path = "./txts/"
    for text_name in text_names:
        path = main_path + text_name + ".txt"
        with open(path, 'r') as f:
            text = f.read()
        n = len(text)
        n = int(n*0.75)
        text = text[-n:]
        text = re.sub("[\.,;~:]"," ",text)
        #text = re.sub("[\[\d()\.,;~*=\]\?:<>/]"," ",text)
        text = re.sub("[\n\t\f\r\v]"," ",text)
        text = re.sub("[ ]{2,}"," ",text)
        text = re.split(" ",text)
        paper_to_refs[text_name] = authors_in_text(text, set_of_authors)
    return paper_to_refs

if __name__ == '__main__':
	text_names = cPickle.load(open('./dfs/text_names100.p', 'rb'))
	set_of_authors = cPickle.load(open('./dfs/set_of_authors_fn100.p', 'rb'))
	paper_to_refs = get_refs(text_names, set_of_authors)
	cPickle.dump(paper_to_refs, open('./dfs/paper_to_refs_fn100.p', 'wb')) 
	
