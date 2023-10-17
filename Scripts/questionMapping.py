import json
import os
import torch
from Scripts.functions import sentenceEmbedding
from tqdm import tqdm

def read_json_files_in_folder(folder_path):
    # List all files in the given folder
    files = os.listdir(folder_path)
    
    # Filter only the JSON files
    json_files = [f for f in files if f.endswith('.json')]
    
    data_list = []
    
    for json_file in json_files:
        with open(os.path.join(folder_path, json_file), 'r') as file:
            data = json.load(file)
            data_list.append(data)
            
    return data_list

def embedding_BigCQ():
    folder_path = 'Data/BigCQ_dataset/query_templates_to_cq_template_mappings/'  # replace with your folder path
    readings = read_json_files_in_folder(folder_path)
    li = [{'Vectors': sentenceEmbedding(r['cqs']), 'query': r['query']} for r in tqdm(readings)]
    torch.save(li, "Data/BigCQ_dataset/BigCQEmbeddings.pt")

def finding_cloests_query(cqEmbedding, vector_sets):
    min_distance = float('inf')
    for index, vector_set in enumerate(vector_sets):
        distances = torch.norm(vector_set['Vectors'] - cqEmbedding, dim=1)
        mean_distance = distances.mean().item()

        if mean_distance < min_distance:
            min_distance = mean_distance
            closest_set_index = index
            mapped_query = vector_sets[closest_set_index]['query']
            
    return mapped_query


def questionMapping(generalizedQuestions, theme_label):
    # Check if the embeddings file already exists, if not, generate embeddings
    if not os.path.exists("Data/Temp/BigCQEmbeddings.pt"):
        print("there is no embedding file for BigCQ, generating one...")
        embedding_BigCQ()
    else:
        print("Already have the embedding file for BigCQ, skipping embedding task...")

    # Load datasets
    BigCQEmbeddings = torch.load("Data/BigCQ_dataset/BigCQEmbeddings.pt")
    for gq in tqdm(generalizedQuestions):
        cq = gq['generalizedPropertyCQ']
        cqEmbedding = sentenceEmbedding(cq) #torch.size([1, 384])
        mapped_query = finding_cloests_query(cqEmbedding, BigCQEmbeddings)
        gq['generalizedPropertyCQ_query'] = mapped_query
        cq = gq['generalizedObjectCQ']
        cqEmbedding = sentenceEmbedding(cq) #torch.size([1, 384])
        mapped_query = finding_cloests_query(cqEmbedding, BigCQEmbeddings)
        gq['generalizedObjectCQ_query'] = mapped_query

    with open(f"Data/Temp/mappedQuestion-{theme_label}.json", 'w') as f:
        json.dump(generalizedQuestions, f, indent=2)
