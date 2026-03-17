from flask import Flask, render_template, request
import requests
import insert_data
#import sql_operations_NC
from bs4 import BeautifulSoup
import re
from collections import defaultdict
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.environ.get("API_KEY")
headers = {
            'authorization': f"Token {api_key}"
        }
def create_app():

    app = Flask(__name__)

    def get_text(idd, searchusr):
        st = ''
        global headers
        url = f'https://api.indiankanoon.org/doc/{idd}/'
        res = requests.post(url,headers=headers).json()

        print("Request for doc with id", idd, "sent")
        try:
            st = res['doc']
            html_string = st
            escaped_string = bytes(html_string, 'utf-8').decode('unicode-escape')    
            soup = BeautifulSoup(escaped_string, "html.parser")
                 
            st = soup.get_text()
        except:
            st = ''
         
        try:
            def get_blockquotes():
                search_strings = [" which reads as", " mutually agreed", " states the following"]
                search_strings.append(str(searchusr))
                soup2 = BeautifulSoup(html_string, 'html.parser')

                filtered_paragraphs = []

                # Find all elements and process them
                elements = soup2.find_all()
                for i, element in enumerate(elements):
                    # Check if the element is a paragraph containing any of the search strings
                    if element.name == 'p' and any(search_string in element.get_text() for search_string in search_strings):
                        # Check the next three elements for <blockquote> elements
                        j = i + 1
                        while j < len(elements) and j <= i + 3:
                            next_element = elements[j]
                            if next_element.name == 'blockquote':
                                filtered_paragraphs.append(next_element.get_text())
                            j += 1
                return filtered_paragraphs
            filtered_paragraphs_lst=get_blockquotes()
            # Combine the values from matching_indents list with newlines between them
            #filtered_paragraphs = '\n'.join(filtered_paragraphs_lst)
            filtered_paragraphs = filtered_paragraphs_lst
        except:
            filtered_paragraphs=''
        return st, filtered_paragraphs

        

    def get_docs(search):
        global headers
        searchusr=search
        S = requests.Session()
        S.headers = headers
        lst = ["clause which reads as", " mutually agreed", "clause states the following"]
        all_data = []
        for qry in lst:
            search = '"' + search + '"' + qry
            search = search.replace(' ', '+') #queries the search text
            for page_num in range(0,3):
                url = f'https://api.indiankanoon.org/search/?formInput={search}&pagenum={page_num}'
                res = S.post(url).json()
                
                if not res['docs']:                              
                    pass
                    #return []
                
                for doc in res['docs']:
                    try:
                        id = doc['tid']
                    except:
                        id = ''
                    try:
                        title = doc['title']
                    except:
                        title = ''
                    try:
                        cleantext, blocktext_lst = get_text(id, searchusr)
                        blocktext = str(blocktext_lst)
                    except:
                        cleantext = ''
                        blocktext = ''
                    try:
                        sze = doc['docsize']
                    except:
                        sze = ''
                    all_data.append(
                        (id,
                        title,
                        cleantext,
                        blocktext,
                        sze)
                    )
        return all_data

    '''st = doc['headline']
    cleantext = re.sub('<.*?>', '', st) #Regex to remove html tags
    cleantext= re.sub(' +', ' ', cleantext) #Regex to remove multi-line spaces'''



    @app.route('/')
    def home():
        return render_template('home.html')


    @app.route('/confirm')
    def shortenurl():
        shortcode=request.args['shortcode']
        lst = get_docs(shortcode)
        with open ("all_data_output.txt", "w") as file:
            file.write(str(lst))
        #ln_lst = len(lst)
        #st_lst = len(set(lst))
        #print(f'found {ln_lst} but because of duplicates {st_lst} are saved to DB')
        results=insert_data.main(lst, shortcode) # lst and shortcode to be passed
        #return '<h1 style="margin: 0 0 30px 0;text-align: center;color: #4bc970;">Data Added!!</h1(results)
        '''
        with open ("Results3.txt", "w") as file:
            file.write(str(results))
        '''
        if results is not None:
            #ln_lst1 = len(results['DocID'])
            ln_lst1 = len(results)
            
        else:
            ln_lst1 = 0
        
        if(ln_lst1!=0):
            return render_template('results.html', results=results, ln_lst=ln_lst1)
        else:
            return render_template('noresults.html')

    return app    

 

 
