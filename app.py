from flask import Flask, render_template, request
import requests
import insert_data
import pipelineoperation
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.environ.get("API_KEY")
headers = {
    'authorization': f"Token {api_key}"
}


def fetch_docmeta(doc_id: str, req_headers: dict) -> dict:
    """Fetch document metadata from Indian Kanoon /docmeta/ endpoint."""
    try:
        url = f'https://api.indiankanoon.org/docmeta/{doc_id}/'
        res = requests.post(url, headers=req_headers).json()
        return {
            'court_name': res.get('court_name', '') or res.get('docsource', ''),
            'judgment_date': res.get('publishdate', '') or res.get('date', ''),
            'case_citation': res.get('citation', '') or res.get('title', ''),
        }
    except Exception:
        return {'court_name': '', 'judgment_date': '', 'case_citation': ''}


def create_app():

    app = Flask(__name__)

    def get_text(idd, searchusr):
        """Fetch a document from IndianKanoon and extract clean text + blockquotes."""
        idd = str(idd)
        st = ''
        html_string = ''
        global headers
        url = f'https://api.indiankanoon.org/doc/{idd}/'
        res = requests.post(url, headers=headers).json()

        print("Request for doc with id", idd, "sent")
        try:
            html_string = res['doc']
            escaped_string = bytes(html_string, 'utf-8').decode('unicode-escape')
            soup = BeautifulSoup(escaped_string, "html.parser")
            st = soup.get_text()
        except Exception:
            st = ''

        try:
            def get_blockquotes():
                search_strings = [
                    "clause", "agreement",
                    " which reads as", " mutually agreed", " states the following",
                ]
                search_strings.append(str(searchusr))
                soup2 = BeautifulSoup(html_string, 'html.parser')
                filtered_paragraphs = []
                elements = soup2.find_all()
                for i, element in enumerate(elements):
                    if element.name == 'p' and any(
                        s in element.get_text() for s in search_strings
                    ):
                        j = i + 1
                        while j < len(elements) and j <= i + 3:
                            if elements[j].name == 'blockquote':
                                filtered_paragraphs.append(elements[j].get_text())
                            j += 1
                return filtered_paragraphs

            filtered_paragraphs = get_blockquotes()
        except Exception:
            filtered_paragraphs = []

        return st, filtered_paragraphs

    def get_docs(search):
        """
        Search IndianKanoon for documents matching the query.
        Returns a dict: {doc_id: {'id': str, 'title': str, 'size': str}}
        """
        global headers
        S = requests.Session()
        S.headers = headers
        query_suffixes = [
            "clause which reads as",
            " mutually agreed",
            "clause states the following",
        ]
        lst_data = {}
        for qry in query_suffixes:
            search_query = ('"' + search + '"' + qry).replace(' ', '+')
            for page_num in range(0, 3):
                url = f'https://api.indiankanoon.org/search/?formInput={search_query}&pagenum={page_num}'
                res = S.post(url).json()
                for doc in res.get('docs', []):
                    doc_id = str(doc.get('tid', ''))
                    if doc_id:
                        if doc_id not in lst_data:
                            lst_data[doc_id] = {'id': doc_id, 'title': '', 'size': '', 'docsource': ''}
                        lst_data[doc_id]['title'] = doc.get('title', '')
                        lst_data[doc_id]['size'] = doc.get('docsize', '')
                        lst_data[doc_id]['docsource'] = doc.get('docsource', '')
        return lst_data

    def get_text_for_new_docs(list_of_docs_not_present, searchusr, lst):
        """
        Fetch full text and blockquotes only for documents not already in the DB.
        Returns a dict: {id: {'id', 'title', 'cleantext', 'blocktext', 'size'}}
        """
        lst_new_data = {}
        for idd in list_of_docs_not_present:
            idd = str(idd)
            lst_new_data[idd] = {
                'id': idd,
                'title': lst.get(idd, {}).get('title', ''),
                'cleantext': '',
                'blocktext': '',
                'size': lst.get(idd, {}).get('size', ''),
            }
            try:
                cleantext, blocktext_lst = get_text(idd, searchusr)
                lst_new_data[idd]['cleantext'] = cleantext
                lst_new_data[idd]['blocktext'] = str(blocktext_lst)
            except Exception:
                pass
        return lst_new_data

    @app.route('/')
    def home():
        return render_template('home.html')

    @app.route('/history')
    def history():
        conn = insert_data.create_connection()
        insert_data.initialize_db(conn)
        searches = insert_data.get_past_searches(conn)
        conn.close()
        return render_template('history.html', searches=searches)

    @app.route('/history/results')
    def history_results():
        query = request.args.get('query', '')
        if not query:
            return render_template('noresults.html')
        conn = insert_data.create_connection()
        insert_data.initialize_db(conn)
        results = insert_data.get_stored_results_for_query(conn, query)
        conn.close()
        if results:
            return render_template(
                'results.html',
                results=results,
                ln_lst=len(results),
                search_query=query,
            )
        return render_template('noresults.html')

    @app.route('/confirm')
    def shortenurl():
        shortcode = request.args.get('shortcode', '')
        if not shortcode:
            return render_template('noresults.html')

        lst = get_docs(shortcode)

        conn = insert_data.create_connection()
        insert_data.initialize_db(conn)

        list_not_present = insert_data.check_for_already_present(conn, lst)
        if list_not_present is None:
            list_not_present = list(lst.keys())

        list_already_present = [d for d in lst if str(d) not in list_not_present]

        lst_new_data = get_text_for_new_docs(list_not_present, shortcode, lst)

        results = insert_data.main(conn, list_already_present, lst_new_data, shortcode)

        if results:
            # Expand snippets to full clause text before classification
            results = insert_data.expand_matched_results(conn, results)

            try:
                results_classified = pipelineoperation.pipeline_operations(results)
                insert_data.add_classified_results(conn, results_classified, shortcode)
            except Exception as e:
                print(f"ML classification failed, falling back to unclassified results: {e}")
                for r in results:
                    r['matching_columns_after_classification'] = r.get('matching_columns', [])
                    r['matching_indents_after_classification'] = r.get('matching_indents', [])
                    r['expanded_columns_after_classification'] = r.get('expanded_columns', [])
                    r['expanded_indents_after_classification'] = r.get('expanded_indents', [])
                results_classified = results

            conn.close()
            return render_template(
                'results.html',
                results=results_classified,
                ln_lst=len(results_classified),
            )

        conn.close()
        return render_template('noresults.html')

    return app
