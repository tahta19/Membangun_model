import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score

# Memuat dataset
data = pd.read_csv('Mall_Customers_Preprocessed.csv')

# Memilih fitur yang relevan
X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Normalisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Set eksperimen MLflow
mlflow.set_experiment("Customer_Segmentation_Tuning")

with mlflow.start_run():
    # Grid parameter untuk hyperparameter tuning
    param_grid = {
        'n_clusters': [3, 4, 5, 6, 7],
        'init': ['k-means++', 'random'],
        'max_iter': [300, 500],
        'n_init': [10, 20]
    }
    
    # Membuat model KMeans
    kmeans = KMeans(random_state=42)
    
    grid_search = GridSearchCV(kmeans, param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X_scaled)

    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    

    mlflow.log_params(best_params)

    mlflow.sklearn.log_model(best_model, "best_model")

    sil_score = silhouette_score(X_scaled, best_model.labels_)
    mlflow.log_metric("silhouette_score", sil_score)


    print(f"Best Parameters: {best_params}")
    print(f"Silhouette Score: {sil_score}")
