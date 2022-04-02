# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 10:41:05 2022

@author: lfurtado
"""
import os
from tqdm import tqdm
from py2neo import Graph
import json
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
# ----------------------- Metadata Inter-Donnés ---------------------

# ## ---------NEO4J

#Connect to the Neo4j Graphj
neo4j_url = "neo4j+s://38996f65.databases.neo4j.io"
user = 'neo4j'
pwd = 'jBsckAJR3eftP-3-XcNGTZCdlm4Snr7WVRZoDGozS-Q'


graph = Graph(neo4j_url, auth=(user, pwd))

#Create the nodes of the graph with the documents id and add the tfidf vectors
ids = []
paths = []
path_to_json = 'enwiki-mini-tfidf/'
for pos_json in tqdm(os.listdir(path_to_json)):
    if pos_json.endswith('.json'):
        path = 'enwiki-mini-tfidf/' + str(pos_json)
        doc = json.load(open(path))
        result = graph.run("""WITH $json AS doc 
        CREATE (document:Document {id: doc.id, vectors: $vectors}) RETURN document
        """, 
            json=doc[0], vectors= doc[0]['vectors'][0]
        )
        ids.append(doc[0]["id"])



# All possible pairs in List
ids_combination = list(combinations(ids, 2))


#Function to get the cossine similarity between vectors
def get_cosine_similarity(feature_vec_1, feature_vec_2):  
    feature_vec_1 = np.array(feature_vec_1)
    feature_vec_2 = np.array(feature_vec_2) 
    return cosine_similarity(feature_vec_1.reshape(1, -1), feature_vec_2.reshape(1, -1))[0][0]

#compare two documents by computing the cossine similarity between them 
#add a relationship between the nodes in the Neo4j graph if the 
#similarity is higher then 0.5
for index, ids in tqdm(enumerate(ids_combination)):
    #get the vectors of each document and compare then by calculating the cossine similarity
    result = list(graph.run(
        """MATCH (d1:Document),(d2:Document) 
        WHERE d1.id = $id1 AND d2.id = $id2
        RETURN d1.vectors, d2.vectors""", 
        id1=ids[0], id2= ids[1]
    ))
    cossine = get_cosine_similarity(result[0]["d1.vectors"], result[0]["d2.vectors"])
    if cossine > 0.5:
        #add the relationship IS_SIMILAR in the Neo4J graph
        graph.run(
         """MATCH (d1:Document),(d2:Document) 
        WHERE d1.id = $id1 AND d2.id = $id2
         CREATE  
        (d1)-[similiar1:IS_SIMILAR{wt:$cossine}]->(d2)""", 
                  id1=ids[0], id2= ids[1], cossine = float(cossine)
         )
