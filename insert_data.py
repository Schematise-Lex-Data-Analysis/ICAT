import sqlite3
from sqlite3 import Error
import re
import json
import pandas as pd
from collections import defaultdict
import ast

regex_pattern = r'^(\"\d+)[\s\S]*?(\"$)'


def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)

    return conn



def create_task(conn, task):
    """
    Create a new task
    :param conn:
    :param task:
    :return:
    """
    conn.commit()
    sql = ''' INSERT OR REPLACE INTO tasks(Doc_ID,Title,Doc_Text,Doc_Blockquotes,Doc_Size)
              VALUES(?,?,?,?,?) '''     #this query is injected to to sql to create a row with headers(doc-id,title,doc_text,docsize) to the table Task
    cur = conn.cursor()   #creates a database connection object
    cur.execute(sql, task) #executes the sql query from above which parses 1 row of data(task) to the table Task
    conn.commit() #as the word mean you are committing the above executation(approving/saving)

    print("Created table with values")
    
    return cur.lastrowid #this for now is negligible to be used for the other task for Indexing

def retrieve_text(conn, query):
    """
    Create a new task
    :param conn:
    :param task:
    :return:
    """
    
    sql_query = "SELECT * FROM tasks"

    # Execute the query to fetch rows from the table
    cursor = conn.cursor()
    cursor.execute(sql_query)

    # Fetch all rows from the cursor
    rows = cursor.fetchall()

        # List to store JSON objects of rows with matching columns
    matching_rows_json = []

    print ("Checking for matching rows")

    for row in rows:
        has_matching_column = False
        has_matching_text = False
        has_matching_indent = False
        matching_columns = {}
        title = None
        doc_id = None
        matching_text = []
        matching_text_with_query = []
        has_matching_indent = False
        matching_indents = []

        for i, column_value in enumerate(row): #generates pairs of (index, column_value) for each cell in the row 
           
            
            if isinstance(column_value, str) and re.search(regex_pattern, column_value, re.MULTILINE | re.DOTALL):
                matching_text = []
                matching_text_with_query = []
                has_matching_text = True
                #matching_text_with_query= find_matching_text_with_query(column_value, query)
                matching_text = find_matching_text(column_value)
                
            if i == 1:  # Replace title_column_index with the index of the "Title" column
                title = column_value
            elif i == 0:  # Replace doc_id_column_index with the index of the "DocID" column
                doc_id = column_value
                
            elif i == 3:
                matching_indent_list=ast.literal_eval(column_value)
                #print("Matching indent list/ list of blockquotes for", title, "is", matching_indent_list)
                if len(matching_indent_list) == 0:
                    has_matching_indent = False
                    #print(title, "has no blockquote")
                else:
                    has_matching_indent = True
                    matching_indents = [value for value in matching_indent_list]
                    #print("Indents for", title, "is", matching_indents)
        if has_matching_text or has_matching_indent:
            row_data = {
                "Title": title,
                "DocID": doc_id,
                "matching_columns": matching_text, #+ matching_text_query, 
                "matching_indents": matching_indents
            }
            matching_rows_json.append(row_data)

    data_dict = matching_rows_json
    # Convert the list of JSON objects to a JSON array
    #json_result = json.dumps(matching_rows_json, indent=2)
    '''
    df = pd.read_json(json_result)

    data_dict = df.to_dict(orient='list')
    '''
    with open("Matching_rows_Format.txt", "w") as file:
        file.write(str(data_dict))

    print("Generated matching rows and file with matching rows")
      
    return data_dict

def delete_sql_records(conn):
    delete_records = "DELETE FROM tasks"

    cur=conn.cursor()
    cur.execute(delete_records)

    print("Deleted records")
    


def find_matching_text_with_query(column_value, query):
    matches = re.finditer(regex_pattern, column_value, re.MULTILINE | re.DOTALL)
    matching_text_with_query = []
    for match in matches:
        if query in match.group():
            matching_text_with_query.append(query)
    print("Matching text with query is: ", matching_text_with_query)
    return matching_text_with_query

def find_matching_text(column_value):
    matching_text = []
    matches = re.finditer(regex_pattern, column_value, re.MULTILINE | re.DOTALL)
    for match in matches:
        matching_text.append(match.group())
    print("Matching text is: ", matching_text)
    return matching_text

def main(lst, query): #lst, query to be added as parameters
    database = "/var/data/pythonsqlite.db"
    
    #query = "arbitration"
    
    # create a database connection
    conn = create_connection(database)
    with conn:
        # create tasks from a list

        delete_sql_records(conn)
        
        for task in lst:
            create_task(conn, task)

        
        
        data_dict=retrieve_text(conn, query)

    
      
    return data_dict
        


if __name__ == '__main__':
    main()
