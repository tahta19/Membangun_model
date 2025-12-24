import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score


data = pd.read_csv('Mall_Customers_Preprocessing.csv')
X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

mlflow.autolog()

mlflow.set_experiment("Customer_Segmentation_Tuning")

with mlflow.start_run():

    param_grid = {
        'n_clusters': [3, 4, 5, 6, 7],
        'init': ['k-means++', 'random'],
        'max_iter': [300, 500],
        'n_init': [10, 20]
    }

    kmeans = KMeans(random_state=42)
    
    grid_search = GridSearchCV(kmeans, param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X_scaled)

    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_


    print(f"Best Parameters: {best_params}")
    
    sil_score = silhouette_score(X_scaled, best_model.labels_)
    print(f"Silhouette Score: {sil_score}")
