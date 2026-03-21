from flask import Flask, render_template, request
import requests
import insert_data
import second_pipelineoperation
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.environ.get("API_KEY")
headers = {
    'authorization': f"Token {api_key}"
}

# Default classification backend from env (.env.example key: CLASSIFIER_BACKEND)
_DEFAULT_CLASSIFIER = os.environ.get("CLASSIFIER_BACKEND", "huggingface").strip().lower()
# Normalise: anything that isn't "regex" maps to "huggingface"
if _DEFAULT_CLASSIFIER not in ("huggingface", "regex"):
    _DEFAULT_CLASSIFIER = "huggingface"

ALL_SUFFIXES = [
    ("clause which reads as", "clause which reads as"),
    (" mutually agreed", "mutually agreed"),
    ("clause states the following", "clause states the following"),
]


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

    def get_docs(search, suffixes=None, page_max=2):
        """
        Search IndianKanoon for documents matching the query.
        Returns a dict: {doc_id: {'id': str, 'title': str, 'size': str}}
        suffixes: list of query suffix strings to use (defaults to all 3)
        page_max: highest page number to fetch (0-2 inclusive)
        """
        global headers
        if suffixes is None:
            suffixes = [s[0] for s in ALL_SUFFIXES]

        S = requests.Session()
        S.headers = headers
        lst_data = {}
        for qry in suffixes:
            search_query = ('"' + search + '"' + qry).replace(' ', '+')
            for page_num in range(0, min(page_max + 1, 3)):
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
        return render_template('dashboard.html', all_suffixes=ALL_SUFFIXES, default_classifier=_DEFAULT_CLASSIFIER)

    @app.route('/history')
    def history():
        conn = insert_data.create_connection()
        if conn is None:
            return render_template('error.html', message="Database connection failed. Please configure DB credentials.")
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
        if conn is None:
            return render_template('error.html', message="Database connection failed.")
        insert_data.initialize_db(conn)
        results = insert_data.get_stored_results_for_query(conn, query)
        conn.close()
        if results:
            return render_template(
                'results.html',
                results=results,
                ln_lst=len(results),
                search_query=query,
                from_history=True,
            )
        return render_template('noresults.html')

    @app.route('/confirm')
    def shortenurl():
        shortcode = request.args.get('shortcode', '')
        if not shortcode:
            return render_template('noresults.html')

        # Parse search options from the dashboard form
        selected_suffixes = request.args.getlist('suffixes')
        if not selected_suffixes:
            selected_suffixes = [s[0] for s in ALL_SUFFIXES]

        try:
            page_max = int(request.args.get('page_max', 2))
            page_max = max(0, min(page_max, 2))
        except (ValueError, TypeError):
            page_max = 2

        classifier = request.args.get('classifier', _DEFAULT_CLASSIFIER)

        lst = get_docs(shortcode, suffixes=selected_suffixes, page_max=page_max)

        if not lst:
            return render_template('noresults.html')

        conn = insert_data.create_connection()
        if conn is None:
            return render_template('error.html', message="Database connection failed. Please configure DB credentials.")
        insert_data.initialize_db(conn)

        list_not_present = insert_data.check_for_already_present(conn, lst)
        if list_not_present is None:
            list_not_present = list(lst.keys())

        list_already_present = [d for d in lst if str(d) not in list_not_present]

        lst_new_data = get_text_for_new_docs(list_not_present, shortcode, lst)

        results = insert_data.main(conn, list_already_present, lst_new_data, shortcode)

        if results:
            results = insert_data.expand_matched_results(conn, results)

            if classifier == 'huggingface':
                try:
                    results_classified = second_pipelineoperation.pipeline_operations(results)

                    # Enrichment: get confidence, reasoning, discussion, sentiment, metadata per result
                    for r in results_classified:
                        # Pick best classified snippet for enrichment
                        best_snippet = None
                        for source in ('expanded_columns_after_classification',
                                       'expanded_indents_after_classification',
                                       'matching_columns_after_classification',
                                       'matching_indents_after_classification'):
                            items = r.get(source, [])
                            if items:
                                best_snippet = items[0]
                                break

                        if best_snippet:
                            clause_text = best_snippet
                            try:
                                ec_result = second_pipelineoperation.expand_and_classify(
                                    conn, r['DocID'], best_snippet)
                                r['classification_confidence'] = ec_result.get('classification_confidence', 0.0)
                                r['classification_reasoning'] = ec_result.get('classification_reasoning', '')
                                r['classification_backend'] = (
                                    'local' if 'local HF' in ec_result.get('classification_reasoning', '')
                                    else 'azure')
                                clause_text = ec_result.get('clause_text', best_snippet)
                            except Exception as e:
                                print(f"expand_and_classify failed for {r['DocID']}: {e}")
                                r['classification_confidence'] = ''
                                r['classification_reasoning'] = ''
                                r['classification_backend'] = 'local'

                            try:
                                disc_result = second_pipelineoperation.extract_discussion_with_azure(
                                    conn, r['DocID'], clause_text)
                                r['extracted_discussion'] = disc_result.get('discussion', '')
                                r['sentiment'] = disc_result.get('sentiment', '')
                                r['sentiment_confidence'] = disc_result.get('sentiment_confidence', 0.0)
                            except Exception as e:
                                print(f"extract_discussion failed for {r['DocID']}: {e}")
                                r['extracted_discussion'] = ''
                                r['sentiment'] = ''
                                r['sentiment_confidence'] = ''
                        else:
                            r['classification_confidence'] = ''
                            r['classification_reasoning'] = ''
                            r['classification_backend'] = 'local'
                            r['extracted_discussion'] = ''
                            r['sentiment'] = ''
                            r['sentiment_confidence'] = ''

                        # Fetch metadata from IndianKanoon API
                        try:
                            meta = second_pipelineoperation.extract_metadata_with_indiankanoon(
                                r['DocID'], headers)
                            r['court_name'] = meta.get('court_name', '')
                            r['judgment_date'] = meta.get('judgment_date', '')
                            r['case_citation'] = meta.get('case_citation', '')
                            insert_data.update_stored_result_metadata(
                                conn, r['DocID'],
                                r['court_name'], r['judgment_date'], r['case_citation'])
                        except Exception as e:
                            print(f"Metadata fetch failed for {r['DocID']}: {e}")
                            r['court_name'] = ''
                            r['judgment_date'] = ''
                            r['case_citation'] = ''

                    insert_data.add_classified_results(conn, results_classified, shortcode)

                except Exception as e:
                    print(f"Classification failed, falling back to regex results: {e}")
                    for r in results:
                        r['matching_columns_after_classification'] = r.get('matching_columns', [])
                        r['matching_indents_after_classification'] = r.get('matching_indents', [])
                        r['expanded_columns_after_classification'] = r.get('expanded_columns', [])
                        r['expanded_indents_after_classification'] = r.get('expanded_indents', [])
                    results_classified = results
            else:
                # Regex-only: no LLM, pass through all matched clauses as-is
                for r in results:
                    r['matching_columns_after_classification'] = r.get('matching_columns', [])
                    r['matching_indents_after_classification'] = r.get('matching_indents', [])
                    r['expanded_columns_after_classification'] = r.get('expanded_columns', [])
                    r['expanded_indents_after_classification'] = r.get('expanded_indents', [])

                    # Still fetch metadata even in regex mode
                    try:
                        meta = second_pipelineoperation.extract_metadata_with_indiankanoon(
                            r['DocID'], headers)
                        r['court_name'] = meta.get('court_name', '')
                        r['judgment_date'] = meta.get('judgment_date', '')
                        r['case_citation'] = meta.get('case_citation', '')
                        insert_data.update_stored_result_metadata(
                            conn, r['DocID'],
                            r['court_name'], r['judgment_date'], r['case_citation'])
                    except Exception:
                        r['court_name'] = ''
                        r['judgment_date'] = ''
                        r['case_citation'] = ''

                results_classified = results
                try:
                    insert_data.add_classified_results(conn, results_classified, shortcode)
                except Exception:
                    pass

            conn.close()
            return render_template(
                'results.html',
                results=results_classified,
                ln_lst=len(results_classified),
                search_query=shortcode,
                classifier=classifier,
                page_max=page_max,
                selected_suffixes=selected_suffixes,
            )

        conn.close()
        return render_template('noresults.html')

    return app
