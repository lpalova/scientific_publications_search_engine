# coding: utf-8

# # Cond-mat.mtrl-sci webscraping

import requests
from bs4 import BeautifulSoup
import cPickle
import os
import subprocess


# ### Scrape first 2000 papers from each year

def get_links_to_papers(main_page):
    response = requests.get(main_page)
    page = response.text
    soup = BeautifulSoup(page,"lxml")
    obj = soup.findAll("span", { "class" : "list-identifier" })
    list_links = []
    for link in obj:
        objlink = link.findAll("a", {"title" : "Download PDF"})
        if len(objlink) > 0:
            l = objlink[0]['href']
            list_links.append(l)
    return list_links

if __name__ == '__main__':
	years = list(range(11,17))
	papers_links = []
	for year in years:
		url = 'http://arxiv.org/list/cond-mat.mtrl-sci/' + str(year) + '?skip=0&show=2000'
		print url
		papers_links.extend(get_links_to_papers(url))
	cPickle.dump(papers_links, open('/home/lucia/05-final-project/source_code/dfs/papers_links.p', 'wb'))
	
	# ### Save pdfs in data folder
	chunk_size = 2000
	for link in papers_links:
		r = requests.get("http://arxiv.org" + link + ".pdf", stream=True)
		input_file = "/home/lucia/05-final-project/source_code/pdfs/" + link[5:] +".pdf"
		with open(input_file, 'wb') as fd:
		    for chunk in r.iter_content(chunk_size):
		        fd.write(chunk)

	# ### Convert pdfs to txts
	main_path = '/home/lucia/05-final-project/source_code/pdfs'
	for input_file in os.listdir(main_path):
		input_path = main_path + input_file
		output_path = "/home/lucia/05-final-project/source_code/txts/" + input_file[:-4] + ".txt"
		args = ["pdftotext", input_path, output_path]
		p = subprocess.Popen(args,  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		o = p.communicate()

