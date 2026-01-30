
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from imblearn.over_sampling import RandomOverSampler
import joblib
import os

# Set MLflow experiment
mlflow.set_experiment("Diabetes_Risk_Prediction")

def load_and_preprocess_data(filepath):
    """Loads data, cleans it, and generates risk labels via clustering."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # 1. Basic Cleaning
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    df = df.drop_duplicates().reset_index(drop=True)
    
    # 2. Handle Zero Values (Treat as missing for clustering)
    # Columns where 0 is invalid
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_cols:
        df[col] = df[col].replace(0, np.nan)

    # 3. Impute and Scale for Clustering
    # We need a temporary set for KMeans
    imputer_cluster = SimpleImputer(strategy='median')
    scaler_cluster = StandardScaler()
    
    X_cluster = df.copy()
    X_cluster[:] = scaler_cluster.fit_transform(imputer_cluster.fit_transform(X_cluster))
    
    # 4. Clustering (KMeans k=2) to generate labels
    # Fix random state for reproducibility of labels
    print("Running KMeans clustering to generate labels...")
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_cluster)
    
    # Identify Risk Categories
    # Cluster 0: Higher Glucose/BMI/Age -> High Risk (1)
    # Cluster 1: Lower values -> Low Risk (0)
    # We check mean Glucose to assign risk label correctly
    mean_glucose = df.groupby('Cluster')['Glucose'].mean()
    if mean_glucose[0] > mean_glucose[1]:
        # Cluster 0 is High Risk
        risk_map = {0: 1, 1: 0}
    else:
        # Cluster 1 is High Risk
        risk_map = {1: 1, 0: 0}
        
    df['risk_category'] = df['Cluster'].map(risk_map)
    print("Risk categories generated.")
    
    return df

def train_model():
    """Main training pipeline with MLflow tracking."""
    
    with mlflow.start_run() as run:
        # Load Data
        df = load_and_preprocess_data("dataset-diabete.csv")
        
        # Features and Target
        feature_cols = ['Glucose', 'BMI', 'DiabetesPedigreeFunction', 'BloodPressure', 
                        'SkinThickness', 'Insulin', 'Age', 'Pregnancies']
        X = df[feature_cols]
        y = df['risk_category']
        
        # Log params
        test_size = 0.2
        random_state = 42
        n_estimators = 200
        max_depth = 10
        min_samples_split = 2
        
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
        
        # Preprocessing Pipeline (Fit on Train)
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        
        X_train = pd.DataFrame(X_train, columns=feature_cols) 
        X_test = pd.DataFrame(X_test, columns=feature_cols)

        # Impute
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)
        
        # Scale
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_test_scaled = scaler.transform(X_test_imp)
        
        # Oversample (Train only)
        ros = RandomOverSampler(random_state=random_state)
        X_train_res, y_train_res = ros.fit_resample(X_train_scaled, y_train)
        
        # Train Model
        print("Training RandomForest Classifier...")
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state
        )
        clf.fit(X_train_res, y_train_res)
        
        # Evaluate
        y_pred = clf.predict(X_test_scaled)
        y_proba = clf.predict_proba(X_test_scaled)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        print(f"Accuracy: {acc}")
        print(f"F1 Score: {f1}")
        print(f"ROC AUC: {roc_auc}")
        
        # Log Metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        
        # Log Model
        mlflow.sklearn.log_model(clf, "model")
        
        # Save Artifacts locally
        print("Saving artifacts...")
        joblib.dump(clf, "final_model.pkl")
        joblib.dump(imputer, "imputer.pkl")
        joblib.dump(scaler, "scaler.pkl")
        joblib.dump(feature_cols, "feature_names.pkl")
        
        # Log Artifacts to MLflow
        mlflow.log_artifact("final_model.pkl")
        mlflow.log_artifact("imputer.pkl")
        mlflow.log_artifact("scaler.pkl")
        mlflow.log_artifact("feature_names.pkl")
        
        print("Training complete. Run tracked in MLflow.")

if __name__ == "__main__":
    train_model()
