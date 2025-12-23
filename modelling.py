import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

data = pd.read_csv('Mall_Customers_Preprocessed.csv')

X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

mlflow.set_experiment("Customer_Segmentation")

with mlflow.start_run():

    kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    
    mlflow.sklearn.log_model(kmeans, "model")
 
    inertia = kmeans.inertia_
    sil_score = silhouette_score(X_scaled, kmeans.labels_)
    
    mlflow.log_metric("inertia", inertia)
    mlflow.log_metric("silhouette_score", sil_score)
    
    print(f"Model trained with inertia: {inertia} and silhouette score: {sil_score}")
