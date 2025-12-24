import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt

data = pd.read_csv('Mall_Customers_Preprocessing.csv')
X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

mlflow.autolog()

mlflow.set_experiment("Customer_Segmentation")

with mlflow.start_run():
    kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_scaled)

    joblib.dump(kmeans, "model.pkl")
    mlflow.log_artifact("model.pkl")
    
    with open('conda.yaml', 'w') as f:
        f.write("name: mlflow-env\nchannels:\n  - defaults\ndependencies:\n  - scikit-learn\n  - pandas\n  - mlflow\n  - pip:\n    - joblib")
    mlflow.log_artifact("conda.yaml")
    
    sil_score = silhouette_score(X_scaled, kmeans.labels_)
    mlflow.log_metric("silhouette_score", sil_score)

    y_true = data['Spending Score (1-100)']
    cm = confusion_matrix(y_true, kmeans.labels_)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(range(len(cm)), range(len(cm)))
    plt.yticks(range(len(cm)), range(len(cm)))
    plt.savefig("training_confusion_matrix.png")
    mlflow.log_artifact("training_confusion_matrix.png")
