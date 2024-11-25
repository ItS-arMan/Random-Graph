## Importing Libraries

```python
import random as r
import networkx as nx
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
```

---

## Feature Extraction


```python
def feature_extraction(graph):
    degrees = [degree for _, degree in graph.degree()]
    degree_prob = np.array(degrees) / sum(degrees)
    entropy = sp.stats.entropy(degree_prob)
    freeman = sum(max(degrees) - degree for degree in degrees) / ((len(graph) - 1) * (len(graph) - 2))

    features = {
        "Number of Nodes": graph.number_of_nodes(),
        "Number of Edges": graph.number_of_edges(),
        "Average Degree": np.mean(degrees),
        "Diameter": nx.diameter(graph),
        "Radius": nx.radius(graph),
        "Density": nx.density(graph),
        "GCC": nx.transitivity(graph),
        "Average Closeness": np.mean(list(nx.closeness_centrality(graph).values())),
        "Average Betweenness": np.mean(list(nx.betweenness_centrality(graph).values())),
        "Average PageRank": np.mean(list(nx.pagerank(graph).values())),
        "Degree Variance": np.var(degrees),
        "Network Entropy": entropy,
        "Freeman Centralization": freeman,
    }
    return features
```

---

## Generating Graph Datasets

Creating three types of random graphs (Erdos-Renyi, Watts-Strogatz, and Barabasi-Albert) and extract their features.

```python
def extract_features_for_graphs(graph_list, model_name):
    features = []
    for i, G in enumerate(graph_list):
        graph_features = feature_extraction(G)
        graph_features["Graph Model"] = f"{model_name}"
        features.append(graph_features)

    return pd.DataFrame(features)

# Generating 150 random graphs
# N: between 300 and 500, p: 0.6
erdos_renyi_graphs = [nx.erdos_renyi_graph(r.randint(300, 500), 0.6) for i in range(50)]

# N: between 300 and 500, p: even numbers between 4 and 20, k: between 0.5 and 0.7
watts_strogatz_graphs = [nx.watts_strogatz_graph(r.randint(300, 500), r.choice(range(4, 20, 2)), r.uniform(0.5, 0.7))
                         for j in range(50)]

# N: between 300 and 500, p: between 10 and 90
barabasi_albert_graphs = [nx.barabasi_albert_graph(r.randint(300, 500), r.randint(10, 90)) for k in range(50)]

erdos_renyi_features = extract_features_for_graphs(erdos_renyi_graphs, "Erdos-Renyi")
watts_strogatz_features = extract_features_for_graphs(watts_strogatz_graphs, "Watts-Strogatz")
barabasi_albert_features = extract_features_for_graphs(barabasi_albert_graphs, "Barabasi-Albert")

all_features = pd.concat([erdos_renyi_features, watts_strogatz_features, barabasi_albert_features], ignore_index=True)

all_features.to_excel("graph_features.xlsx", index=False)
```

---

## Applying Machine Learning

### K-Nearest Neighbors (KNN)

Standardizeing the features and evaluate KNN with three distance metrics: **Euclidean**, **Manhattan**, and **Cosine**.

```python
stand_features = StandardScaler().fit_transform(all_features.iloc[:, :-1].values)
labels = all_features["Graph Model"].array
encoded_labels = LabelEncoder().fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(
    stand_features, encoded_labels, test_size=0.1, random_state=42, stratify=encoded_labels
)

metrics = ['euclidean', 'manhattan', 'cosine']

for metric in metrics:
    knn = KNeighborsClassifier(n_neighbors=3, metric=metric)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    scores = cross_val_score(knn, stand_features, encoded_labels, cv=5)
    print(f"
Using {metric} distance:")
    print(f"Cross-Validation Accuracy: {scores.mean():.2f}")
    print(f"Misclassified Samples: {sum(y_test != y_pred) / len(y_pred):.2f}")
```
#### Results
-- Using euclidean distance:
Cross-Validation Accuracy: 1.00
Number of misclassified samples (euclidean): 0.0

-- Using manhattan distance:
Cross-Validation Accuracy: 1.00
Number of misclassified samples (manhattan): 0.0

-- Using cosine distance:
Cross-Validation Accuracy: 1.00
Number of misclassified samples (cosine): 0.0


---

### Decision Tree Classifier

Feature importance is calculated to identify key graph characteristics.

```python
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

importance = pd.DataFrame({
    'Feature': all_features.columns[:-1],
    'Importance': clf.feature_importances_,
}).sort_values(by="Importance", ascending=False)

print("
Decision Tree Feature Importance:
", importance)
```
#### Results 
| Feature                  | Importance |
|--------------------------|------------|
| Number of Nodes          | 0.000000   |
| Number of Edges          | 0.000000   |
| Average Degree           | 0.000000   |
| Diameter                  | 0.000000   |
| Radius                  | 0.000000   |
| Density                 | 0.501818   |
| GCC                     | 0.000000   |
| Average Closeness       | 0.498182   |
| Average Betweenness     | 0.000000   |
| Average PageRank        | 0.000000   |
| Degree Variance         | 0.000000   |
| Network Entropy         | 0.000000   |
| Freeman Centralization  | 0.000000   |
