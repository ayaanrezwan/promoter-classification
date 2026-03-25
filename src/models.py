"""
models.py — Model Training, Evaluation, and Comparison Pipeline

This module handles:
1. Train/test splitting with stratification
2. Training Logistic Regression, SVM, and Random Forest
3. Cross-validation for robust performance estimates
4. Hyperparameter tuning via GridSearchCV
5. Evaluation metrics (accuracy, precision, recall, F1, ROC-AUC)
6. Feature importance extraction from all three models

Author: Ayaonic
Project: Human Promoter Classification
"""

import numpy as np
import os
import json
from sklearn.model_selection import (
    train_test_split, cross_validate, StratifiedKFold, GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.pipeline import Pipeline


# =============================================================================
# SECTION 1: DATA SPLITTING
# =============================================================================

def prepare_data(X, labels, test_size=0.2, random_state=42):
    """
    Split data into training and test sets with stratification.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    print(f"Data split (test_size={test_size}):")
    print(f"  Training:  {X_train.shape[0]} samples "
          f"({np.mean(y_train):.3f} positive rate)")
    print(f"  Testing:   {X_test.shape[0]} samples "
          f"({np.mean(y_test):.3f} positive rate)")

    return X_train, X_test, y_train, y_test


# =============================================================================
# SECTION 2: MODEL DEFINITIONS
# =============================================================================

def get_models():
    """
    Return a dictionary of model names to (pipeline, hyperparameter_grid) pairs.
    """
    models = {}

    models['Logistic Regression'] = {
        'pipeline': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=5000, random_state=42))
        ]),
        'param_grid': {
            'clf__C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'clf__penalty': ['l1', 'l2'],
            'clf__solver': ['liblinear']
        }
    }

    models['SVM'] = {
        'pipeline': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(probability=True, random_state=42))
        ]),
        'param_grid': {
            'clf__C': [0.1, 1.0, 10.0],
            'clf__kernel': ['linear', 'rbf'],
            'clf__gamma': ['scale', 'auto']
        }
    }

    models['Random Forest'] = {
        'pipeline': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(random_state=42, n_jobs=-1))
        ]),
        'param_grid': {
            'clf__n_estimators': [100, 200, 500],
            'clf__max_depth': [10, 20, 30, None],
            'clf__min_samples_split': [2, 5, 10],
            'clf__max_features': ['sqrt', 'log2']
        }
    }

    return models


# =============================================================================
# SECTION 3: CROSS-VALIDATION
# =============================================================================

def cross_validate_models(X_train, y_train, models=None, cv_folds=5):
    """
    Run stratified k-fold cross-validation on all models with default
    hyperparameters to establish baseline performance.
    """
    if models is None:
        models = get_models()

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    cv_results = {}

    print("=" * 60)
    print(f"CROSS-VALIDATION (Stratified {cv_folds}-Fold, Default Hyperparameters)")
    print("=" * 60)

    for name, model_config in models.items():
        print(f"\n--- {name} ---")
        pipeline = model_config['pipeline']

        scores = cross_validate(
            pipeline, X_train, y_train,
            cv=cv, scoring=scoring,
            return_train_score=False, n_jobs=-1
        )

        result = {}
        for metric in scoring:
            values = scores[f'test_{metric}']
            mean_val = values.mean()
            std_val = values.std()
            result[metric] = (mean_val, std_val)
            print(f"  {metric:>12}: {mean_val:.4f} +/- {std_val:.4f}")

        cv_results[name] = result

    return cv_results


# =============================================================================
# SECTION 4: HYPERPARAMETER TUNING
# =============================================================================

def tune_models(X_train, y_train, models=None, cv_folds=5):
    """
    Tune hyperparameters for all models using GridSearchCV.
    """
    if models is None:
        models = get_models()

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    best_models = {}

    print("=" * 60)
    print(f"HYPERPARAMETER TUNING (GridSearchCV, {cv_folds}-Fold)")
    print("=" * 60)

    for name, model_config in models.items():
        print(f"\n--- Tuning {name} ---")
        pipeline = model_config['pipeline']
        param_grid = model_config['param_grid']

        n_combos = 1
        for values in param_grid.values():
            n_combos *= len(values)
        print(f"  Searching {n_combos} combinations x {cv_folds} folds "
              f"= {n_combos * cv_folds} fits")

        grid_search = GridSearchCV(
            pipeline, param_grid,
            cv=cv, scoring='f1',
            n_jobs=-1, return_train_score=False, verbose=0
        )
        grid_search.fit(X_train, y_train)

        print(f"  Best F1 score: {grid_search.best_score_:.4f}")
        print(f"  Best parameters:")
        for param, value in grid_search.best_params_.items():
            clean_name = param.replace('clf__', '')
            print(f"    {clean_name}: {value}")

        best_models[name] = grid_search

    return best_models


# =============================================================================
# SECTION 5: FINAL EVALUATION ON TEST SET
# =============================================================================

def evaluate_on_test_set(best_models, X_test, y_test):
    """
    Evaluate tuned models on the held-out test set. This is called ONCE.
    """
    test_results = {}
    roc_data = {}

    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON HELD-OUT TEST SET")
    print("=" * 60)

    for name, grid_search in best_models.items():
        print(f"\n--- {name} ---")
        best_pipeline = grid_search.best_estimator_

        y_pred = best_pipeline.predict(X_test)
        y_prob = best_pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob)
        }

        for metric_name, value in metrics.items():
            print(f"  {metric_name:>12}: {value:.4f}")

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        print(f"\n  Confusion Matrix:")
        print(f"    Predicted:   Non-Prom  Promoter")
        print(f"    Actual NP:   {tn:>7}   {fp:>7}")
        print(f"    Actual P:    {fn:>7}   {tp:>7}")
        print(f"\n{classification_report(y_test, y_pred, target_names=['Non-promoter', 'Promoter'])}")

        test_results[name] = metrics
        test_results[name]['confusion_matrix'] = cm

        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr, thresholds)

    # Print comparison table
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<22} {'Accuracy':>9} {'Precision':>10} "
          f"{'Recall':>8} {'F1':>8} {'ROC-AUC':>9}")
    print("  " + "-" * 68)
    for name, metrics in test_results.items():
        print(f"  {name:<22} {metrics['accuracy']:>9.4f} "
              f"{metrics['precision']:>10.4f} "
              f"{metrics['recall']:>8.4f} {metrics['f1']:>8.4f} "
              f"{metrics['roc_auc']:>9.4f}")

    return test_results, roc_data


# =============================================================================
# SECTION 6: FEATURE IMPORTANCE
# =============================================================================

def extract_feature_importance(best_models, vocabulary):
    """
    Extract feature importance from all three models.
    """
    importance_dict = {}

    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)

    for name, grid_search in best_models.items():
        best_pipeline = grid_search.best_estimator_
        clf = best_pipeline.named_steps['clf']

        print(f"\n--- {name} ---")

        if name == 'Logistic Regression':
            importances = clf.coef_.squeeze()
            importance_type = "coefficient"
        elif name == 'SVM':
            if clf.kernel == 'linear':
                importances = clf.coef_.squeeze()
                importance_type = "coefficient"
            else:
                print(f"  SVM with '{clf.kernel}' kernel — no direct feature importances.")
                importances = np.zeros(len(vocabulary))
                importance_type = "not available (non-linear kernel)"
        elif name == 'Random Forest':
            importances = clf.feature_importances_
            importance_type = "Gini importance"
        else:
            continue

        kmer_importance = list(zip(vocabulary, importances))
        kmer_importance.sort(key=lambda x: abs(x[1]), reverse=True)

        print(f"  Importance type: {importance_type}")
        print(f"  Top 15 most important k-mers:")
        print(f"    {'K-mer':<8} {'Importance':>12} {'Direction':<20}")
        print(f"    {'-' * 42}")
        for kmer, imp in kmer_importance[:15]:
            if name == 'Random Forest':
                direction = f"(Gini: {imp:.5f})"
            else:
                direction = "-> promoter" if imp > 0 else "-> non-promoter"
            print(f"    {kmer:<8} {imp:>+12.5f} {direction:<20}")

        importance_dict[name] = kmer_importance

    return importance_dict


# =============================================================================
# SECTION 7: SAVE RESULTS
# =============================================================================

def save_results(test_results, importance_dict, output_dir='results/metrics'):
    """
    Save evaluation metrics and feature importances to disk.
    """
    os.makedirs(output_dir, exist_ok=True)

    metrics_to_save = {}
    for name, metrics in test_results.items():
        metrics_to_save[name] = {
            k: v for k, v in metrics.items()
            if k != 'confusion_matrix'
        }

    metrics_path = os.path.join(output_dir, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)

    for name, importances in importance_dict.items():
        safe_name = name.lower().replace(' ', '_')
        imp_path = os.path.join(output_dir, f'feature_importance_{safe_name}.csv')
        with open(imp_path, 'w') as f:
            f.write('kmer,importance\n')
            for kmer, imp in importances:
                f.write(f'{kmer},{imp:.8f}\n')

    print(f"\nResults saved to {output_dir}/")


# =============================================================================
# SECTION 8: MAIN PIPELINE
# =============================================================================

def run_full_pipeline(X, labels, vocabulary):
    """
    Execute the complete model training and evaluation pipeline.
    """
    # Step 1: Split
    X_train, X_test, y_train, y_test = prepare_data(X, labels)

    # Step 2: Baseline cross-validation
    print("\n" + "#" * 60)
    print("# PHASE 1: BASELINE CROSS-VALIDATION")
    print("#" * 60)
    cv_results = cross_validate_models(X_train, y_train)

    # Step 3: Hyperparameter tuning
    print("\n" + "#" * 60)
    print("# PHASE 2: HYPERPARAMETER TUNING")
    print("#" * 60)
    best_models = tune_models(X_train, y_train)

    # Step 4: Test set evaluation
    print("\n" + "#" * 60)
    print("# PHASE 3: TEST SET EVALUATION")
    print("#" * 60)
    test_results, roc_data = evaluate_on_test_set(best_models, X_test, y_test)

    # Step 5: Feature importance
    print("\n" + "#" * 60)
    print("# PHASE 4: FEATURE IMPORTANCE")
    print("#" * 60)
    importance_dict = extract_feature_importance(best_models, vocabulary)

    # Step 6: Save everything
    save_results(test_results, importance_dict)

    return {
        'cv_results': cv_results,
        'best_models': best_models,
        'test_results': test_results,
        'roc_data': roc_data,
        'importance_dict': importance_dict,
        'splits': (X_train, X_test, y_train, y_test)
    }
