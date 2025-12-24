import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt

data = pd.read_csv('Mall_Customers_Preprocessing.csv')
X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

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

    mlflow.log_param("best_n_clusters", best_params['n_clusters'])
    mlflow.log_param("best_init", best_params['init'])
    
    joblib.dump(best_model, "best_model.pkl")
    mlflow.log_artifact("best_model.pkl")

    with open('requirements.txt', 'w') as f:
        f.write("scikit-learn==0.24.1\npandas==1.2.3\nmlflow==1.19.0\njoblib==1.0.1")
    mlflow.log_artifact("requirements.txt")

    sil_score = silhouette_score(X_scaled, best_model.labels_)
    mlflow.log_metric("silhouette_score", sil_score)

    y_true = data['Spending Score (1-100)']
    cm = confusion_matrix(y_true, best_model.labels_)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(range(len(cm)), range(len(cm)))
    plt.yticks(range(len(cm)), range(len(cm)))
    plt.savefig("best_model_confusion_matrix.png")
    mlflow.log_artifact("best_model_confusion_matrix.png")
