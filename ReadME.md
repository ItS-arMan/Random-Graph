
# Graph Analysis and Machine Learning Pipeline

This document outlines the steps to analyze graph structures and apply machine learning techniques using Python.

---

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

The `feature_extraction` function calculates key graph metrics like average degree, density, and centralization measures.

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

We create three types of random graphs (Erdős-Rényi, Watts-Strogatz, and Barabási-Albert) and extract their features.

```python
def extract_features_for_graphs(graph_list, model_name):
    features = []
    for G in graph_list:
        graph_features = feature_extraction(G)
        graph_features["Graph Model"] = model_name
        features.append(graph_features)
    return pd.DataFrame(features)

# Generate 50 graphs for each model
erdos_renyi_graphs = [nx.erdos_renyi_graph(r.randint(300, 500), 0.6) for _ in range(50)]
watts_strogatz_graphs = [nx.watts_strogatz_graph(r.randint(300, 500), r.choice(range(4, 20, 2)), r.uniform(0.5, 0.7)) for _ in range(50)]
barabasi_albert_graphs = [nx.barabasi_albert_graph(r.randint(300, 500), r.randint(10, 90)) for _ in range(50)]

# Extract features
erdos_renyi_features = extract_features_for_graphs(erdos_renyi_graphs, "Erdos-Renyi")
watts_strogatz_features = extract_features_for_graphs(watts_strogatz_graphs, "Watts-Strogatz")
barabasi_albert_features = extract_features_for_graphs(barabasi_albert_graphs, "Barabasi-Albert")

# Combine all features
all_features = pd.concat([erdos_renyi_features, watts_strogatz_features, barabasi_albert_features], ignore_index=True)

# Save to Excel
all_features.to_excel("graph_features.xlsx", index=False)
```

---

## Applying Machine Learning

### K-Nearest Neighbors (KNN)

We standardize the features and evaluate KNN with three distance metrics: **Euclidean**, **Manhattan**, and **Cosine**.

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

---

This pipeline provides a foundation for analyzing graph data and applying machine learning models to classify graph structures.
