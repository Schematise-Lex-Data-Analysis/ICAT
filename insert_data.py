import re
import ast
import psycopg
from datetime import datetime
import pytz
from dotenv import load_dotenv
import os

load_dotenv()

regex_pattern = r'^(\"\d+)[\s\S]*?(\"$)'

CLAUSE_BOUNDARY_PATTERN = re.compile(
    r'\n\s*(?:\d+\.\s|\(\d+\)\s|\([a-z]\)\s|\([ivxlc]+\)\s|(?:Sub-?)?[Cc]lause\s|CLAUSE\s|[Aa]rticle\s|ARTICLE\s|[Ss]ection\s|SECTION\s|[Ss]chedule\s|SCHEDULE\s)',
    re.IGNORECASE
)

DB_HOST = os.environ.get("DB_HOST")
DB_NAME = os.environ.get("DB_NAME")
DB_USER = os.environ.get("DB_USER")
DB_PASS = os.environ.get("DB_PASS")
DB_SSLMODE = os.environ.get("SSLMODE", "require")


def create_connection():
    """Create a database connection using environment variables."""
    conn = None
    try:
        db_uri = f"host={DB_HOST} dbname={DB_NAME} user={DB_USER} password={DB_PASS} sslmode={DB_SSLMODE}"
        conn = psycopg.connect(db_uri)
    except Exception as e:
        print("Error during connection:\n", e)
    return conn


def initialize_db(conn):
    """Create all required tables if they do not already exist."""
    with conn.cursor() as cur:
        cur.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                Doc_ID  TEXT PRIMARY KEY,
                Title   TEXT,
                Doc_Text        TEXT,
                Doc_Blockquotes TEXT,
                Doc_Size        TEXT
            )
        ''')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS stored_results (
                Doc_ID  TEXT PRIMARY KEY,
                Title   TEXT,
                Doc_Text        TEXT,
                Doc_Blockquotes TEXT,
                Doc_Size        TEXT
            )
        ''')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS classified_index (
                Doc_Id  TEXT,
                Title   TEXT,
                searchquery     TEXT,
                matching_indents        TEXT,
                matching_columns        TEXT,
                matching_columns_after_classification TEXT,
                matching_indents_after_classification TEXT,
                expanded_columns TEXT,
                expanded_indents TEXT,
                expanded_columns_after_classification TEXT,
                expanded_indents_after_classification TEXT
            )
        ''')
        # Add columns for existing databases that don't have them yet
        for col in ['expanded_columns', 'expanded_indents',
                     'expanded_columns_after_classification',
                     'expanded_indents_after_classification']:
            try:
                cur.execute(f'ALTER TABLE classified_index ADD COLUMN {col} TEXT')
            except Exception:
                conn.rollback()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS search_queries (
                searchquery TEXT,
                dateandtime TEXT
            )
        ''')
        conn.commit()
    print("Database initialized")


def check_for_already_present(conn, dict_of_docs):
    """Return the list of doc IDs from dict_of_docs that are NOT in stored_results."""
    list_of_docs = [str(k) for k in dict_of_docs.keys()]
    try:
        result = []
        with conn.cursor() as cur:
            for docid in list_of_docs:
                cur.execute("SELECT Doc_ID FROM stored_results WHERE Doc_ID = %s", (docid,))
                fetch_result = cur.fetchone()
                if fetch_result:
                    result.append(str(fetch_result[0]))

        set_not_present = set(list_of_docs) - set(result)
        return list(set_not_present)

    except Exception as error:
        print(f"Database operation failed: {error}")
        return None


def create_task(conn, task):
    """
    Insert or update a document in both 'tasks' and 'stored_results'.
    :param task: list/tuple of [Doc_ID, Title, Doc_Text, Doc_Blockquotes, Doc_Size]
    """
    sql_task = '''
        INSERT INTO tasks(Doc_ID, Title, Doc_Text, Doc_Blockquotes, Doc_Size)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (Doc_ID) DO UPDATE SET
            Title           = EXCLUDED.Title,
            Doc_Text        = EXCLUDED.Doc_Text,
            Doc_Blockquotes = EXCLUDED.Doc_Blockquotes,
            Doc_Size        = EXCLUDED.Doc_Size;
    '''
    sql_stored = '''
        INSERT INTO stored_results(Doc_ID, Title, Doc_Text, Doc_Blockquotes, Doc_Size)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (Doc_ID) DO UPDATE SET
            Title           = EXCLUDED.Title,
            Doc_Text        = EXCLUDED.Doc_Text,
            Doc_Blockquotes = EXCLUDED.Doc_Blockquotes,
            Doc_Size        = EXCLUDED.Doc_Size;
    '''
    try:
        with conn.cursor() as cur:
            cur.execute(sql_task, task)
            cur.execute(sql_stored, task)
            conn.commit()
    except Exception as e:
        print(f"An error occurred in create_task: {e}")
        conn.rollback()
    finally:
        print("create_task() completed.")


def add_stored_results(conn, list_of_ids):
    """Copy rows from stored_results → tasks for already-seen doc IDs."""
    sql = '''
        INSERT INTO tasks (Doc_ID, Title, Doc_Text, Doc_Blockquotes, Doc_Size)
        SELECT Doc_ID, Title, Doc_Text, Doc_Blockquotes, Doc_Size
        FROM stored_results
        WHERE Doc_ID = %s
        ON CONFLICT (Doc_ID) DO NOTHING;
    '''
    try:
        with conn.cursor() as cur:
            for docid in list_of_ids:
                cur.execute(sql, (str(docid),))
            conn.commit()
        print("Stored data transferred to tasks.")
    except Exception as e:
        print(f"An error occurred in add_stored_results: {e}")
        conn.rollback()
    finally:
        print("add_stored_results() completed.")


def delete_sql_records(conn):
    """Clear the tasks table for the current search session."""
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM tasks")
            conn.commit()
        print("Deleted records from tasks.")
    except Exception as e:
        print(f"Error deleting records: {e}")


def retrieve_text(conn, query):
    """Scan tasks table and return rows matching the regex pattern or having blockquotes."""
    sql_query = "SELECT * FROM tasks"
    matching_rows_json = []

    try:
        with conn.cursor() as cur:
            cur.execute(sql_query)
            rows = cur.fetchall()
    except Exception as e:
        print(f"Warning: could not retrieve from tasks table: {e}")
        return []

    print("Checking for matching rows")

    for row in rows:
        has_matching_text = False
        has_matching_indent = False
        title = None
        doc_id = None
        matching_text = []
        matching_indents = []

        for i, column_value in enumerate(row):
            if isinstance(column_value, str) and re.search(regex_pattern, column_value, re.MULTILINE | re.DOTALL):
                has_matching_text = True
                matching_text = find_matching_text(column_value)

            if i == 1:
                title = column_value
            elif i == 0:
                doc_id = column_value
            elif i == 3:
                try:
                    matching_indent_list = ast.literal_eval(column_value)
                    if matching_indent_list:
                        has_matching_indent = True
                        matching_indents = list(matching_indent_list)
                except Exception:
                    pass

        if has_matching_text or has_matching_indent:
            matching_rows_json.append({
                "Title": title,
                "DocID": doc_id,
                "matching_columns": matching_text,
                "matching_indents": matching_indents,
            })

    with open("Matching_rows_Format.txt", "w") as file:
        file.write(str(matching_rows_json))

    print("Generated matching rows")
    return matching_rows_json


def add_classified_results(conn, dict_of_results, searchquery):
    """Persist ML-classified results to classified_index and log the search query."""
    sql_classified = '''
        INSERT INTO classified_index(
            Doc_Id, Title, searchquery,
            matching_indents, matching_columns,
            matching_columns_after_classification,
            matching_indents_after_classification,
            expanded_columns, expanded_indents,
            expanded_columns_after_classification,
            expanded_indents_after_classification
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    '''
    sql_search = '''
        INSERT INTO search_queries(searchquery, dateandtime) VALUES (%s, %s)
    '''
    ist = pytz.timezone('Asia/Kolkata')
    current_datetime_ist = datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S')

    try:
        with conn.cursor() as cur:
            for result in dict_of_results:
                cur.execute(sql_classified, (
                    result['DocID'],
                    result['Title'],
                    searchquery,
                    str(result.get('matching_indents', [])),
                    str(result.get('matching_columns', [])),
                    str(result.get('matching_columns_after_classification', [])),
                    str(result.get('matching_indents_after_classification', [])),
                    str(result.get('expanded_columns', [])),
                    str(result.get('expanded_indents', [])),
                    str(result.get('expanded_columns_after_classification', [])),
                    str(result.get('expanded_indents_after_classification', [])),
                ))
            cur.execute(sql_search, (searchquery, current_datetime_ist))
            conn.commit()
    except Exception as e:
        print(f"Error adding classified results: {e}")
        conn.rollback()


def _find_snippet_position(doc_text, snippet):
    """Find the character position of a snippet within the full document text."""
    if not doc_text or not snippet:
        return -1
    # Normalize whitespace in both strings
    normalized_doc = re.sub(r'\s+', ' ', doc_text)
    normalized_snippet = re.sub(r'\s+', ' ', snippet)
    pos = normalized_doc.find(normalized_snippet)
    if pos != -1:
        return pos
    # Fallback: build a regex from first ~100 chars of snippet
    prefix = normalized_snippet[:100]
    try:
        pattern = re.sub(r'\s+', r'\\s+', re.escape(prefix))
        match = re.search(pattern, normalized_doc)
        if match:
            return match.start()
    except re.error:
        pass
    return -1


def _find_clause_boundaries(doc_text, match_pos, context_before=2000, context_after=3000):
    """Find clause start and end boundaries around a match position."""
    # Scan backward for clause boundary
    search_start = max(0, match_pos - context_before)
    before_text = doc_text[search_start:match_pos]

    clause_start = search_start
    # Look for clause boundary markers scanning backward
    boundaries = list(CLAUSE_BOUNDARY_PATTERN.finditer(before_text))
    if boundaries:
        # Use the last (nearest) boundary before the match
        clause_start = search_start + boundaries[-1].start()
    else:
        # Fall back to double newline
        double_nl = before_text.rfind('\n\n')
        if double_nl != -1:
            clause_start = search_start + double_nl

    # Scan forward for clause boundary
    search_end = min(len(doc_text), match_pos + context_after)
    after_text = doc_text[match_pos:search_end]

    clause_end = search_end
    boundary_match = CLAUSE_BOUNDARY_PATTERN.search(after_text[1:])  # skip current position
    if boundary_match:
        clause_end = match_pos + 1 + boundary_match.start()
    else:
        double_nl = after_text.find('\n\n', 1)
        if double_nl != -1:
            clause_end = match_pos + double_nl

    # Hard cap at 5000 chars
    if clause_end - clause_start > 5000:
        clause_end = clause_start + 5000

    return clause_start, clause_end


def extract_full_clause(doc_text, indicator_snippet):
    """Extract the full clause surrounding a matched snippet from the document text."""
    if not doc_text or not indicator_snippet:
        return indicator_snippet
    pos = _find_snippet_position(doc_text, indicator_snippet)
    if pos == -1:
        return indicator_snippet
    start, end = _find_clause_boundaries(doc_text, pos)
    return doc_text[start:end].strip()


def expand_matched_results(conn, results):
    """Expand short indicator snippets to full clause text using the document text from DB."""
    # Collect all DocIDs we need
    doc_ids = [r['DocID'] for r in results if r.get('DocID')]
    if not doc_ids:
        return results

    # Batch-fetch Doc_Text from tasks table
    doc_texts = {}
    try:
        with conn.cursor() as cur:
            for doc_id in doc_ids:
                cur.execute("SELECT Doc_Text FROM tasks WHERE Doc_ID = %s", (doc_id,))
                row = cur.fetchone()
                if row and row[0]:
                    doc_texts[doc_id] = row[0]
    except Exception as e:
        print(f"Warning: could not fetch Doc_Text for expansion: {e}")
        # Return results with empty expanded fields as fallback
        for r in results:
            r['expanded_columns'] = r.get('matching_columns', [])
            r['expanded_indents'] = r.get('matching_indents', [])
        return results

    for r in results:
        doc_text = doc_texts.get(r.get('DocID'), '')

        expanded_columns = []
        for snippet in r.get('matching_columns', []):
            expanded_columns.append(extract_full_clause(doc_text, snippet))
        r['expanded_columns'] = expanded_columns

        expanded_indents = []
        for snippet in r.get('matching_indents', []):
            expanded_indents.append(extract_full_clause(doc_text, snippet))
        r['expanded_indents'] = expanded_indents

    print("Expanded matched results to full clause text")
    return results


def get_past_searches(conn):
    """Return all past search queries ordered by most recent first."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT searchquery, dateandtime FROM search_queries ORDER BY dateandtime DESC"
            )
            return cur.fetchall()
    except Exception as e:
        print(f"Error fetching past searches: {e}")
        return []


def get_stored_results_for_query(conn, searchquery):
    """Fetch classified_index rows for a given past search query."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT Doc_Id, Title, matching_indents, matching_columns, "
                "matching_columns_after_classification, matching_indents_after_classification, "
                "expanded_columns, expanded_indents, "
                "expanded_columns_after_classification, expanded_indents_after_classification "
                "FROM classified_index WHERE searchquery = %s",
                (searchquery,),
            )
            rows = cur.fetchall()
    except Exception as e:
        print(f"Error fetching stored results: {e}")
        return []

    results = []
    for row in rows:
        results.append({
            'DocID': row[0],
            'Title': row[1],
            'matching_indents': _safe_eval(row[2]),
            'matching_columns': _safe_eval(row[3]),
            'matching_columns_after_classification': _safe_eval(row[4]),
            'matching_indents_after_classification': _safe_eval(row[5]),
            'expanded_columns': _safe_eval(row[6]),
            'expanded_indents': _safe_eval(row[7]),
            'expanded_columns_after_classification': _safe_eval(row[8]),
            'expanded_indents_after_classification': _safe_eval(row[9]),
        })
    return results


def _safe_eval(val):
    """Safely parse a stringified list back into a Python list.

    Handles both Python format (['a', 'b']) and PostgreSQL array format ({"a","b"}).
    """
    if not val:
        return []
    val = val.strip()
    # PostgreSQL array format: starts and ends with braces
    if val.startswith('{') and val.endswith('}'):
        inner = val[1:-1].strip()
        if not inner:
            return []
        # Extract double-quoted strings, honouring \" escapes inside them
        items = re.findall(r'"((?:[^"\\]|\\.)*)"', inner)
        if items:
            # Unescape \" → " and \\ → \
            return [i.replace('\\"', '"').replace('\\\\', '\\') for i in items]
        # Fallback: unquoted elements (e.g. {NULL} or {word,word})
        return [i.strip() for i in inner.split(',') if i.strip() and i.strip() != 'NULL']
    # Python list format
    try:
        result = ast.literal_eval(val)
        return result if isinstance(result, list) else []
    except Exception:
        return []


def find_matching_text_with_query(column_value, query):
    matches = re.finditer(regex_pattern, column_value, re.MULTILINE | re.DOTALL)
    matching_text_with_query = []
    for match in matches:
        if query in match.group():
            matching_text_with_query.append(query)
    return matching_text_with_query


def find_matching_text(column_value):
    matching_text = []
    matches = re.finditer(regex_pattern, column_value, re.MULTILINE | re.DOTALL)
    for match in matches:
        matching_text.append(match.group())
    return matching_text


def main(conn, list_of_docs_already_present, lst_new_data, query):
    """
    Orchestrate a search session:
    1. Clear tasks table.
    2. Re-populate from stored_results for already-seen docs.
    3. Insert new docs (writing to both tasks and stored_results).
    4. Return regex/blockquote-matched results.
    """
    initialize_db(conn)
    delete_sql_records(conn)
    add_stored_results(conn, list_of_docs_already_present)

    values_list = [list(subdict.values()) for subdict in lst_new_data.values()]
    for task in values_list:
        create_task(conn, task)

    return retrieve_text(conn, query)


if __name__ == '__main__':
    main(create_connection(), [], {}, '')
