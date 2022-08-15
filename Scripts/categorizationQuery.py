# install qwikidata
!pip install qwikidata

from qwikidata.sparql import return_sparql_query_results

# send any sparql query to the wikidata query service and get full result back

NE = ["University of Tirana", "University of Bologna", "Rudolf Avenhaus"]

for i in NE:
    sparql_query = f"""
        SELECT DISTINCT ?cLabel WHERE {{
          ?Q wdt:P31/wdt:P279? ?c .
          ?c rdfs:label ?cLabel .
          ?Q rdfs:label "{i}"@en .
        }} 
        LIMIT 1
        """
    res = return_sparql_query_results(sparql_query)
    print(res)
