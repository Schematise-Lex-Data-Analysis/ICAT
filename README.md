# NonCompeteTestRelease
An sqlite3 version of a contract clause search on IndianKanoon, through it's API

Currently, the database is matching all the contractual clauses stored in it, ever (irrespective of query)

We need to make sure the Doc_ID's returned from search are stored in a list/SQL table, which will then check if DocID's are already stored, and store them in a separate temporary table for processing further. 

TODO - 18th September 
0) Create query embedding for blockquotes
1) Create Table for Blockquotes
2) Create table for queries and specific results which have been requested more than 'n' times on the app. Treat this as a cached database.
3) Caching - put a counter for the number of requests against each DocID
4) Build view from document + search accessed

EXECUTION - PSEUDOCODE

1) Make a request to the stored texts, to see if DocID is available.
2) Make a request to the blockquotes table, to see if the same query has been made before.
3) Get blockquotes and put in the table if not there along with query name.
4) Store the blockquote and query in a view which will be dropped after it has been passed.
