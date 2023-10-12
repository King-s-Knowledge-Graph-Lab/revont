import random
import collections
from functools import partial

import umap
import hdbscan

import numpy as np
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm.notebook import trange
from hyperopt import fmin, tpe, STATUS_OK, space_eval, Trials
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from typing import List

import hashlib
import logging
from sentence_transformers import SentenceTransformer, util, LoggingHandler

def preprocess_questions(raw_questions : List[str]):
  """Clean strings and filter out same questions"""
  # Basic string clearning operations
  new_questions = [q.strip() for q in raw_questions]
  # Removing duplicates already in the set (trivial)
  new_questions = list(set(new_questions))

  return new_questions


def generate_clusters(question_embeddings : List[str],
                      n_neighbors : int,
                      n_components : int,
                      min_cluster_size : int,
                      random_state : int = None):
    """
    Generate HDBSCAN cluster object after reducing embedding dimensionality with UMAP
    """

    umap_reduced = (umap.UMAP(n_neighbors=n_neighbors,
                              n_components=n_components,
                              metric='cosine',
                              random_state=random_state)
                            .fit_transform(question_embeddings))

    clusters = hdbscan.HDBSCAN(min_cluster_size = min_cluster_size,
                               metric='euclidean',
                               cluster_selection_method='eom').fit(umap_reduced)

    return clusters, umap_reduced


def plot_best_clusters(embeddings, cluster_labels, n_neighbors=15, min_dist=0.1,
                       figsize=(14, 8)):
        """
        Reduce dimensionality of best clusters and plot in 2D using instance
        variable result of running bayesian_search
        Arguments:
            n_neighbors: float, UMAP hyperparameter n_neighbors
            min_dist: float, UMAP hyperparameter min_dist for effective
                      minimum distance between embedded points
        """

        umap_reduce = (umap.UMAP(n_neighbors=n_neighbors,
                                 n_components=2,
                                 min_dist=min_dist,
                                 # metric='cosine',
                                 random_state=42)
                           .fit_transform(embeddings)
                       )

        point_size = 100.0 / np.sqrt(embeddings.shape[0])

        result = pd.DataFrame(umap_reduce, columns=['x', 'y'])
        result['labels'] = cluster_labels

        fig, ax = plt.subplots(figsize=figsize)
        noise = result[result.labels == -1]
        clustered = result[result.labels != -1]
        plt.scatter(noise.x, noise.y, color='lightgrey', s=point_size)
        plt.scatter(clustered.x, clustered.y, c=clustered.labels,
                    s=point_size, cmap='jet')
        plt.colorbar()
        plt.show()

def EvalDataLoading(selected_model):
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])


    # Choose a model for the encodings for now
    model = selected_model

    hasher = hashlib.sha256  # hash is used to store pairs efficiently
    get_hash = lambda x: hasher(x.encode('utf-8')).hexdigest()

    qpp_data = pd.read_csv("Data/qqp.tsv", sep="\t")
    qpp_data["question1"] = qpp_data["question1"].astype(str)
    qpp_data["question2"] = qpp_data["question2"].astype(str)

    all_questions = list(set(list(qpp_data.question1) + list(qpp_data.question2)))
    print(f"There are {len(all_questions)} unique questions in QPP data")

    all_questions_hash = [get_hash(str(q)) for q in all_questions]

    positive_hashes, negative_hashes = set(), set()

    for i, qpair in qpp_data[["question1", "question2", "is_duplicate"]].iterrows():
        q1_hash = get_hash(qpair["question1"])
        q2_hash = get_hash(qpair["question2"])

    if qpair["is_duplicate"] == 0:
        negative_hashes.add((q1_hash, q2_hash))
        # negative_hashes.add((q2_hash, q1_hash))
    else:  # assuming them to be positive then
        positive_hashes.add((q1_hash, q2_hash))
        # positive_hashes.add((q2_hash, q1_hash))

    # Start the multi-process pool on all available CUDA devices
    pool = model.start_multi_process_pool()

    questions_embeddings = model.encode_multi_process(all_questions, pool)

    # Optional: Stop the proccesses in the pool
    model.stop_multi_process_pool(pool)

    n_neighbors = 15
    n_components = 5
    min_cluster_size = 2

    clusters = generate_clusters(question_embeddings=questions_embeddings,
                                n_neighbors=n_neighbors,
                                n_components=n_components,
                                min_cluster_size=min_cluster_size,
                                random_state=42)

    label_count = len(np.unique(clusters.labels_))

    print(f"Found {label_count} unique cluesters")

def CQClustering(question):
    print(f"Starting no. of questions: {len(questions)}")
    new_questions = preprocess_questions(questions)
    print(f"No. of questions after pre-processing: {len(new_questions)}")
    questions_sbert = model.encode(new_questions)
    questions_sbert.shape
    n_neighbors = 15
    n_components = 5
    min_cluster_size = 2

    clusters, reduced_embed = \
    generate_clusters(question_embeddings=questions_sbert,
                    n_neighbors=n_neighbors,
                    n_components=n_components,
                    min_cluster_size=min_cluster_size,
                    random_state=SEED)
    cluster_labels = clusters.labels_
    label_count = len(np.unique(cluster_labels))
    total_num = len(clusters.labels_)
    print(f"Found {label_count} unique cluesters")
    plot_best_clusters(questions_sbert, cluster_labels, figsize=(25, 10))

    question_assignments = pd.DataFrame({"question": new_questions, "cluster": cluster_labels})
    question_assignments.sort_values("cluster", inplace=True, ascending=False)
    return question_assignments

def ParaphraseDetection(qpp_data, model):
    all_questions = list(set(list(qpp_data.question1) + list(qpp_data.question2)))
    print(f"There are {len(all_questions)} unique questions in QPP data")
    x, y = qpp_data[["question1", "question2"]].values, qpp_data["is_duplicate"].values
    #Start the multi-process pool on all available CUDA devices
    pool = model.start_multi_process_pool()

    #Compute the embeddings using the multi-process pool
    x_a = model.encode_multi_process(x[:, 0], pool)
    x_b = model.encode_multi_process(x[:, 1], pool)

    #Optional: Stop the proccesses in the pool
    model.stop_multi_process_pool(pool)
    x_a_sample = torch.tensor(x_a[:20]).to(torch.float32)
    x_b_sample = torch.tensor(x_b[:20]).to(torch.float32)

    x_a_t = torch.tensor(x_a).to(torch.float32)
    x_b_t = torch.tensor(x_b).to(torch.float32)



if __name__ == '__main__':
    # Option 1: Clustering-based question filtering
    with open('questions.txt') as f:
        questions = f.readlines()
    clustering_results = CQClustering(questions)

    # Option 2: Similar paraphrase detection based on the pre-trained model
    #Candidate model list
    model_st1 = SentenceTransformer('all-mpnet-base-v2')
    model_st2 = SentenceTransformer('all-MiniLM-L6-v2')
    model_st3 = SentenceTransformer('paraphrase-mpnet-base-v2')
    model_st4 = SentenceTransformer('paraphrase-MiniLM-L3-v2')

    qpp_data = pd.read_csv("Data/qqp.tsv", sep="\t")
    ParaphraseDetection(qpp_data, model_st2)


    EvalDataLoading(model_st2)