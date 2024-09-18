
from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question')
    # 在此处实现你的对话逻辑，例如返回一个简单的响应
    response = {"answer": "This is a placeholder answer."}
    return jsonify(response)


if __name__ == '__main__':
    app.run()


import pandas as pd
import numpy as np
from modelscope import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
import random

csv_file = "questions.csv"
questions_df = pd.read_csv(csv_file)

model_dir = "path_to_your_model_directory"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

def tokenize_and_count(text):
    tokens = tokenizer.tokenize(text)
    return dict((token, tokens.count(token)) for token in set(tokens))

def vectorize(token_counts, vocabulary):
    vector = [token_counts.get(token, 0) for token in vocabulary]
    return vector

token_counts_list = sql_questions_df['问题'].apply(tokenize_and_count).tolist()
all_tokens = set(token for token_counts in token_counts_list for token in token_counts)
vocabulary = list(all_tokens)
```python
将词频转换为向量
```python
vectors = np.array([vectorize(token_counts, vocabulary) for token_counts in token_counts_list])

similarity_matrix = cosine_similarity(vectors)


def find_best_cluster_number(similarity_matrix, max_clusters=10):
    silhouette_scores = []
    for n_clusters in range(2, max_clusters + 1):
        clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='kmeans')
        cluster_labels = clustering.fit_predict(similarity_matrix)
        silhouette_avg = silhouette_score(similarity_matrix, cluster_labels, metric="precomputed")
        silhouette_scores.append((n_clusters, silhouette_avg))

    best_n_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
    return best_n_clusters

best_num_clusters = find_best_cluster_number(similarity_matrix, max_clusters=20)
print(f"最佳聚类数: {best_num_clusters}")
clustering = SpectralClustering(n_clusters=best_num_clusters, affinity='precomputed', assign_labels='kmeans')
labels = clustering.fit_predict(similarity_matrix)
sql_questions_df['Cluster'] = labels
sampled_questions = []
for cluster in range(best_num_clusters):
    cluster_questions = sql_questions_df[sql_questions_df['Cluster'] == cluster]
    if len(cluster_questions) > 2:
        sampled_questions.extend(cluster_questions.sample(n=2).to_dict('records'))
    else:
        sampled_questions.extend(cluster_questions.to_dict('records'))
for sample in sampled_questions:
    print(f"问题id: {sample['问题id']}, 问题: {sample['问题']}")
sampled_df = pd.DataFrame(sampled_questions)
sampled_df.to_csv("sampled_questions.csv", index=False)
