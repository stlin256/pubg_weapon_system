import os
import argparse
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from . import config

def get_model(model_name, use_cuda=False):
    """
    根据模型名称返回一个模型实例.
    """
    if model_name == 'RandomForest':
        return RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_STATE, n_jobs=-1)
    elif model_name == 'KNN':
        return KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    elif model_name == 'SVM':
        return SVC(kernel='rbf', C=1.0, random_state=config.RANDOM_STATE, probability=True)
    elif model_name == 'SVM_GridSearch':
        # Pipeline to scale data and then apply SVM
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf', random_state=config.RANDOM_STATE, probability=True))
        ])
        
        # Parameter grid for GridSearch
        param_grid = {
            'svm__C': [0.1, 1, 10],
            'svm__gamma': ['scale', 'auto']
        }
        
        # GridSearchCV object
        grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=2)
        return grid_search
    elif model_name == 'XGBoost':
        params = {
            'objective': 'multi:softmax',
            'eval_metric': 'mlogloss',
            'eta': 0.1,
            'max_depth': 3,
            'seed': config.RANDOM_STATE
        }
        if use_cuda:
            print("Configuring XGBoost to use CUDA.")
            # Using 'hist' as a robust fallback. For full GPU support,
            # XGBoost needs to be compiled with CUDA support.
            # 'gpu_hist' is the ideal value if supported.
            params['tree_method'] = 'hist'
            params['device'] = 'cuda'
        return xgb.XGBClassifier(**params)
    elif model_name == 'LightGBM':
        params = {
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'n_estimators': 200,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'max_depth': -1,
            'seed': config.RANDOM_STATE,
            'n_jobs': -1
        }
        if use_cuda:
            print("Configuring LightGBM to use CUDA.")
            params['device'] = 'gpu'
        return lgb.LGBMClassifier(**params)
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")

def main():
    parser = argparse.ArgumentParser(description="Train a specified model on a specified target.")
    parser.add_argument('--model', type=str, required=True, help="The model to train (e.g., 'RandomForest', 'XGBoost').")
    parser.add_argument('--target', type=str, required=True, help="The target to predict (e.g., 'weapon', 'distance', 'direction', or 'all' for multi-output).")
    parser.add_argument('--use_cuda', action='store_true', help="Use CUDA for GPU acceleration if available.")
    parser.add_argument('--cross_validate', action='store_true', help="Perform 5-fold cross-validation instead of training a final model.")
    args = parser.parse_args()

    print(f"--- Processing model '{args.model}' for target '{args.target}' ---")

    # --- 1. 加载特征数据 ---
    features_path = os.path.join(config.FEATURES_PATH, 'train_features.npy')
    X_train = np.load(features_path)
    nsamples, nx, ny = X_train.shape
    X_train_reshaped = X_train.reshape((nsamples, nx * ny))
    
    # --- 2. 加载标签数据 ---
    if args.target == 'all':
        y_weapon = np.load(os.path.join(config.FEATURES_PATH, 'train_labels_weapon.npy'))
        y_distance = np.load(os.path.join(config.FEATURES_PATH, 'train_labels_distance.npy'))
        y_direction = np.load(os.path.join(config.FEATURES_PATH, 'train_labels_direction.npy'))
        y_train = np.vstack([y_weapon, y_distance, y_direction]).T
        print(f"Loaded all targets for multi-output. Labels shape: {y_train.shape}")
    else:
        labels_path = os.path.join(config.FEATURES_PATH, f'train_labels_{args.target}.npy')
        y_train = np.load(labels_path)
        print(f"Loaded training data. Features shape: {X_train_reshaped.shape}, Labels shape: {y_train.shape}")

    # --- 3. 获取模型 ---
    base_model = get_model(args.model, args.use_cuda)
    
    if args.target == 'all':
        model = MultiOutputClassifier(base_model, n_jobs=-1)
    else:
        model = base_model

    # --- 4. 执行交叉验证或最终训练 ---
    if args.cross_validate:
        print("--- Performing 5-fold cross-validation ---")
        scores = cross_val_score(model, X_train_reshaped, y_train, cv=5, scoring='accuracy', n_jobs=-1)
        print(f"Cross-validation scores: {scores}")
        print(f"Average score: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
        print("Cross-validation finished. No model was saved.")
    else:
        print("--- Training final model on the entire training set ---")
        print("Starting model training...")
        model.fit(X_train_reshaped, y_train)
        print("Model training complete.")
        
        train_accuracy = model.score(X_train_reshaped, y_train)
        print(f"Accuracy (score) on training set: {train_accuracy:.4f}")
        
        model_filename = f'{args.model}_{args.target}.pkl'
        model_save_path = os.path.join(config.MODEL_SAVE_PATH, model_filename)
        joblib.dump(model, model_save_path)
        print(f"Model saved to {model_save_path}")

if __name__ == '__main__':
    main()