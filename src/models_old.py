"""
models.py — Model Training, Evaluation, and Comparison Pipeline

This module handles:
1. Train/test splitting with stratification
2. Training three classifiers: Logistic Regression, SVM, Random Forest
3. Hyperparameter tuning via GridSearchCV
4. Evaluation metrics: accuracy, precision, recall, F1, ROC-AUC
5. Cross-validation for robust performance estimates
6. Model comparison framework

Author: Ayaonic
Project: Human Promoter Classification
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    cross_validate
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve
)
import warnings
warnings.filterwarnings('ignore')  # Suppress convergence warnings during grid search


# =============================================================================
# SECTION 1: DATA SPLITTING
# =============================================================================

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and test sets with stratification.

    Stratification means: the class ratio in the split matches the
    class ratio in the full dataset. If 50% of our data is promoter,
    then 50% of the training set AND 50% of the test set will be promoter.

    Without stratification, random splits can accidentally put 60% promoters
    in training and 40% in test (or worse), which biases your evaluation.
    With only 2 balanced classes this is unlikely to be extreme, but
    stratification is good practice regardless — and essential when classes
    are imbalanced.

    Why 80/20?
    ----------
    This is a common default. 80% gives the model enough data to learn
    patterns. 20% gives you enough test samples for reliable metric
    estimates. Other common splits are 70/30 and 90/10.

    For small datasets (<1000 samples): use more test data (70/30)
    For large datasets (>100k samples): can use less (90/10 or 95/5)
    Our dataset (~32k samples) is medium-sized — 80/20 is ideal.

    Parameters
    ----------
    X : numpy array, shape (n_samples, n_features)
    y : numpy array, shape (n_samples,)
    test_size : float
        Fraction of data reserved for testing.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test : numpy arrays
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # THIS is the key argument — ensures balanced splits
    )

    print(f"Data split: {X_train.shape[0]} train / {X_test.shape[0]} test")
    print(f"  Train class balance: {y_train.mean():.3f} "
          f"({np.sum(y_train==1)} pos / {np.sum(y_train==0)} neg)")
    print(f"  Test class balance:  {y_test.mean():.3f} "
          f"({np.sum(y_test==1)} pos / {np.sum(y_test==0)} neg)")

    return X_train, X_test, y_train, y_test


# =============================================================================
# SECTION 2: FEATURE SCALING
# =============================================================================

def scale_features(X_train, X_test):
    """
    Standardize features to zero mean and unit variance.

    For each feature (column), this computes:
        X_scaled = (X - mean) / std

    After scaling, each feature has mean ≈ 0 and std ≈ 1.

    Why scale?
    ----------
    Logistic Regression and SVM are sensitive to feature magnitudes.

    Consider two features: feature A ranges [0, 0.001] and feature B
    ranges [0, 0.5]. Without scaling, the model's weight for feature A
    would need to be ~500x larger than B's weight to have equal influence.
    The optimizer struggles with this because gradients are dominated by
    the large-scale features.

    Scaling puts all features on the same playing field, so the optimizer
    converges faster and the resulting weights are more interpretable.

    Random Forest does NOT need scaling because it uses decision thresholds
    on individual features — the magnitude doesn't affect where the split
    point lands. We scale anyway for consistency across models.

    CRITICAL: Fit the scaler on TRAINING data only
    -----------------------------------------------
    The scaler learns mean and std from the training set. We then APPLY
    those same statistics to transform the test set. We NEVER fit on
    test data because that would leak information about the test
    distribution into our preprocessing.

    This is a form of data leakage that's easy to miss:
    - WRONG:  scaler.fit(X_all), then split
    - WRONG:  scaler.fit(X_train), scaler.fit(X_test)  (fits test too!)
    - RIGHT:  scaler.fit(X_train), scaler.transform(X_test)

    Parameters
    ----------
    X_train, X_test : numpy arrays

    Returns
    -------
    X_train_scaled, X_test_scaled : numpy arrays
    scaler : fitted StandardScaler object (keep this for later use)
    """
    scaler = StandardScaler()

    # fit_transform = fit (learn mean/std) + transform (apply), in one call
    X_train_scaled = scaler.fit_transform(X_train)

    # transform only — uses the mean/std learned from training data
    X_test_scaled = scaler.transform(X_test)

    print(f"Feature scaling applied:")
    print(f"  Train mean range: [{X_train_scaled.mean(axis=0).min():.4f}, "
          f"{X_train_scaled.mean(axis=0).max():.4f}]  (should be ~0)")
    print(f"  Train std range:  [{X_train_scaled.std(axis=0).min():.4f}, "
          f"{X_train_scaled.std(axis=0).max():.4f}]  (should be ~1)")

    return X_train_scaled, X_test_scaled, scaler


# =============================================================================
# SECTION 3: MODEL DEFINITIONS
# =============================================================================

def get_models():
    """
    Return a dictionary of model names → (model_instance, param_grid) tuples.

    Each model has a hyperparameter grid for tuning. Here's what each
    hyperparameter controls:

    LOGISTIC REGRESSION
    -------------------
    C : Inverse regularization strength.
        - Small C (e.g. 0.01) = strong regularization = simpler model
        - Large C (e.g. 100)  = weak regularization  = complex model
        Regularization prevents overfitting by penalizing large weights.
        The loss function becomes: Loss = error + (1/C) * ||weights||

        Think of it this way: C controls how much the model is "allowed"
        to fit the training data. Low C says "keep it simple even if you
        miss some training examples." High C says "try to get every
        training example right, even if that means complex decision
        boundaries."

    penalty : Type of regularization.
        - 'l1' (Lasso): Can zero out weights entirely → automatic feature
          selection. Some k-mer weights become exactly 0.
        - 'l2' (Ridge): Shrinks all weights toward zero but never exactly
          to zero. Default and usually sufficient.

    SVM (Support Vector Machine)
    ----------------------------
    C : Same concept as logistic regression — tradeoff between margin
        width and classification errors. Large C = narrow margin, fewer
        misclassifications on training data.

    kernel : The function that computes similarity between data points.
        - 'linear': Decision boundary is a hyperplane. Like logistic
          regression but optimizes for maximum margin.
        - 'rbf' (Radial Basis Function): Maps data to infinite-dimensional
          space. Can learn non-linear boundaries. If RBF >> linear, your
          data has important non-linear structure.

    gamma : (only for RBF kernel) Controls the "reach" of each training
        example's influence.
        - Small gamma = each point influences a large area = smoother boundary
        - Large gamma = each point only influences nearby points = wiggly boundary
        - 'scale' = 1 / (n_features * X.var()), a data-dependent default

    RANDOM FOREST
    -------------
    n_estimators : Number of trees in the forest. More trees = more stable
        predictions (variance reduction) but slower training. There's no
        overfitting risk from adding more trees — each tree is independent.
        Diminishing returns above ~200-500 trees typically.

    max_depth : Maximum depth of each tree.
        - None = trees grow until all leaves are pure (can overfit)
        - 10 = each tree makes at most 10 sequential decisions
        - Shallow trees = high bias, low variance (underfitting)
        - Deep trees = low bias, high variance (overfitting)

    min_samples_split : Minimum samples required to split a node.
        - 2 (default) = can split even if only 2 samples remain
        - 10 = needs at least 10 samples to consider splitting
        - Higher values = more regularization

    max_features : Number of features considered at each split.
        - 'sqrt' = √(n_features) ≈ 8 for our 64 features
        - This randomization is KEY to why random forests work.
          Each tree sees a different random subset of features, so
          trees make different errors, and averaging them cancels out
          the errors (ensemble effect).
    """
    models = {
        'Logistic Regression': (
            LogisticRegression(
                max_iter=5000,       # Ensure convergence
                random_state=42,
                solver='saga',       # Supports both l1 and l2 penalty
            ),
            {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
            }
        ),

        'SVM': (
            SVC(
                random_state=42,
                probability=True,    # Needed for ROC-AUC (adds computation)
            ),
            {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto'],  # Only used with rbf
            }
        ),

        'Random Forest': (
            RandomForestClassifier(
                random_state=42,
                n_jobs=-1,           # Use all CPU cores for parallel tree building
            ),
            {
                'n_estimators': [100, 200, 500],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'max_features': ['sqrt', 'log2'],
            }
        ),
    }

    return models


# =============================================================================
# SECTION 4: HYPERPARAMETER TUNING WITH CROSS-VALIDATION
# =============================================================================

def tune_model(model, param_grid, X_train, y_train, cv_folds=5):
    """
    Find the best hyperparameters using GridSearchCV.

    How GridSearchCV works:
    ----------------------
    It tries EVERY combination of hyperparameters in the grid.
    For each combination, it runs k-fold cross-validation:

    1. Split training data into k folds
    2. For each fold:
       - Train on k-1 folds
       - Evaluate on the held-out fold
    3. Average the scores across all folds
    4. The combination with the highest average score wins

    Example with our Logistic Regression grid:
    - C has 5 values, penalty has 2 values → 10 combinations
    - Each combination is evaluated with 5-fold CV → 50 total fits
    - The best (C, penalty) pair is selected

    Why cross-validation instead of a single validation split?
    ----------------------------------------------------------
    A single 80/20 split gives you ONE score that depends on which
    specific samples ended up in the validation set. With CV, you get
    k scores from k different splits, and the mean is a more reliable
    estimate of true performance.

    scoring='roc_auc': We optimize for ROC-AUC rather than accuracy.
    AUC measures the model's ability to RANK positive examples higher
    than negatives across all possible thresholds. It's more informative
    than accuracy for classification because:
    - Accuracy depends on a specific threshold (0.5 by default)
    - AUC evaluates performance across ALL thresholds
    - AUC is unaffected by class imbalance (though ours is balanced)

    Parameters
    ----------
    model : sklearn estimator instance
    param_grid : dict of parameter → list of values
    X_train, y_train : training data
    cv_folds : int, number of cross-validation folds

    Returns
    -------
    best_model : fitted model with best hyperparameters
    cv_results : dict with detailed cross-validation results
    """
    # StratifiedKFold ensures each fold has the same class ratio
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,          # Parallelize across CPU cores
        verbose=0,          # Suppress per-fold output (too noisy)
        return_train_score=True  # Also track training scores (for overfitting detection)
    )

    print(f"  Running grid search...")
    n_combinations = 1
    for values in param_grid.values():
        n_combinations *= len(values)
    print(f"  {n_combinations} parameter combinations x {cv_folds} folds "
          f"= {n_combinations * cv_folds} total fits")

    grid_search.fit(X_train, y_train)

    print(f"  Best parameters: {grid_search.best_params_}")
    print(f"  Best CV ROC-AUC: {grid_search.best_score_:.4f}")

    # Check for overfitting: if train score >> test score, the model
    # is memorizing training data rather than learning general patterns
    best_idx = grid_search.best_index_
    train_score = grid_search.cv_results_['mean_train_score'][best_idx]
    test_score = grid_search.cv_results_['mean_test_score'][best_idx]
    print(f"  Train vs CV score: {train_score:.4f} vs {test_score:.4f} "
          f"(gap: {train_score - test_score:.4f})")

    if train_score - test_score > 0.05:
        print(f"  WARNING: Possible overfitting (gap > 0.05)")

    return grid_search.best_estimator_, grid_search.cv_results_


# =============================================================================
# SECTION 5: MODEL EVALUATION
# =============================================================================

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Compute comprehensive evaluation metrics on the test set.

    Metrics explained:
    -----------------
    Accuracy:   (TP + TN) / Total
                Fraction of all predictions that were correct.
                Intuitive but misleading for imbalanced classes.

    Precision:  TP / (TP + FP)
                "Of everything the model called a promoter, what fraction
                actually IS a promoter?"
                High precision = few false alarms.

    Recall:     TP / (TP + FN)
                "Of all actual promoters, what fraction did the model find?"
                High recall = few missed promoters.
                Also called "sensitivity" or "true positive rate."

    F1 Score:   2 * (Precision * Recall) / (Precision + Recall)
                The harmonic mean of precision and recall. Balances both.
                F1 = 1.0 is perfect. F1 = 0.5 means the model is struggling.
                We use the harmonic mean (not arithmetic) because it
                penalizes extreme imbalances: if precision=1.0 and recall=0.1,
                arithmetic mean = 0.55 (looks okay), harmonic mean = 0.18
                (correctly reflects the poor recall).

    ROC-AUC:    Area Under the Receiver Operating Characteristic curve.
                Measures the probability that the model ranks a random
                positive example higher than a random negative example.
                AUC = 0.5 means random guessing. AUC = 1.0 is perfect.
                Threshold-independent — evaluates all possible thresholds.

    Confusion Matrix:
                            Predicted
                         Neg      Pos
        Actual  Neg  [  TN   |   FP  ]
                Pos  [  FN   |   TP  ]

    TP = True Positive:  Model said promoter, it IS a promoter (correct)
    TN = True Negative:  Model said non-promoter, it IS non-promoter (correct)
    FP = False Positive: Model said promoter, but it's NOT (Type I error)
    FN = False Negative: Model said non-promoter, but it IS one (Type II error)

    Parameters
    ----------
    model : fitted sklearn estimator
    X_test, y_test : test data
    model_name : str, for display purposes

    Returns
    -------
    metrics : dict with all computed metrics
    y_pred : predicted labels
    y_prob : predicted probabilities (for ROC curve plotting)
    """
    # Get predictions
    y_pred = model.predict(X_test)

    # Get probability scores for the positive class
    # predict_proba returns [[P(class=0), P(class=1)], ...] for each sample
    # We take column 1 → P(class=1) = P(promoter)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
    }

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  {model_name} — Test Set Results")
    print(f"{'='*50}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Neg    Pos")
    print(f"  Actual Neg  [{cm[0,0]:>5}  {cm[0,1]:>5}]")
    print(f"  Actual Pos  [{cm[1,0]:>5}  {cm[1,1]:>5}]")

    # Compute ROC curve points (for plotting later)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    metrics['fpr'] = fpr
    metrics['tpr'] = tpr

    return metrics, y_pred, y_prob


# =============================================================================
# SECTION 6: FULL PIPELINE — TRAIN AND COMPARE ALL MODELS
# =============================================================================

def train_and_compare(X, y, test_size=0.2, cv_folds=5):
    """
    The main entry point: splits data, tunes all models, evaluates them,
    and returns a comparison summary.

    This function chains together the entire Phase 3 pipeline:
    1. Split → 2. Scale → 3. Tune each model → 4. Evaluate → 5. Compare

    Parameters
    ----------
    X : numpy array, shape (n_samples, n_features)
    y : numpy array, shape (n_samples,)
    test_size : float
    cv_folds : int

    Returns
    -------
    results : dict
        Keys are model names, values are dicts containing:
        'model': fitted model, 'metrics': evaluation metrics,
        'y_pred': predictions, 'y_prob': probabilities,
        'cv_results': cross-validation details
    comparison_df : pandas DataFrame
        Side-by-side comparison of all models' metrics.
    scaler : fitted StandardScaler
    """
    # Step 1: Split
    print("=" * 60)
    print("PHASE 3: MODEL TRAINING AND EVALUATION")
    print("=" * 60)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size)

    # Step 2: Scale
    print(f"\n{'='*60}")
    print("FEATURE SCALING")
    print("=" * 60)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Step 3: Get model definitions
    models = get_models()

    # Step 4: Tune and evaluate each model
    results = {}
    for name, (model, param_grid) in models.items():
        print(f"\n{'='*60}")
        print(f"TRAINING: {name}")
        print(f"{'='*60}")

        # Tune hyperparameters
        best_model, cv_results = tune_model(
            model, param_grid, X_train_scaled, y_train, cv_folds
        )

        # Evaluate on test set
        metrics, y_pred, y_prob = evaluate_model(
            best_model, X_test_scaled, y_test, name
        )

        results[name] = {
            'model': best_model,
            'metrics': metrics,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'cv_results': cv_results,
        }

    # Step 5: Build comparison table
    print(f"\n{'='*60}")
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)

    comparison_data = []
    for name, result in results.items():
        m = result['metrics']
        comparison_data.append({
            'Model': name,
            'Accuracy': f"{m['accuracy']:.4f}",
            'Precision': f"{m['precision']:.4f}",
            'Recall': f"{m['recall']:.4f}",
            'F1': f"{m['f1']:.4f}",
            'ROC-AUC': f"{m['roc_auc']:.4f}",
        })

    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))

    # Identify the best model
    best_name = max(results.keys(),
                    key=lambda k: results[k]['metrics']['roc_auc'])
    print(f"\n  Best model by ROC-AUC: {best_name} "
          f"({results[best_name]['metrics']['roc_auc']:.4f})")

    return results, comparison_df, scaler, (X_train_scaled, X_test_scaled,
                                             y_train, y_test)


# =============================================================================
# SECTION 7: FEATURE IMPORTANCE EXTRACTION
# =============================================================================

def get_feature_importance(results, vocabulary):
    """
    Extract feature importance from each model type.

    Each model provides importance differently:

    Logistic Regression → coefficients (model.coef_)
        The weight for each feature in the linear equation:
        log(P(y=1)/P(y=0)) = w0 + w1*x1 + w2*x2 + ... + w64*x64

        Positive coefficient = this k-mer increases P(promoter)
        Negative coefficient = this k-mer decreases P(promoter)
        Magnitude = strength of influence

    SVM (linear kernel) → coefficients (model.coef_)
        Same interpretation as logistic regression.
        For RBF kernel, we can't extract direct feature weights because
        the model operates in a transformed (possibly infinite-dimensional)
        space. We'll use permutation importance as a fallback.

    Random Forest → impurity-based importance (model.feature_importances_)
        For each feature, this measures the total reduction in Gini impurity
        across all trees in the forest when that feature is used for splitting.

        Gini impurity measures how "mixed" a node is:
        Gini = 1 - P(class0)^2 - P(class1)^2
        Pure node (all one class): Gini = 0
        Maximally mixed (50/50): Gini = 0.5

        A feature that creates very pure splits has high importance.

    Parameters
    ----------
    results : dict from train_and_compare
    vocabulary : list of k-mer strings

    Returns
    -------
    importance_df : pandas DataFrame with importance scores per model
    """
    importance_data = {'kmer': vocabulary}

    for name, result in results.items():
        model = result['model']

        if name == 'Logistic Regression':
            # coef_ has shape (1, n_features) for binary classification
            # We take [0] to get the 1D array
            importance_data['LR_coef'] = model.coef_[0]

        elif name == 'SVM':
            if model.kernel == 'linear':
                importance_data['SVM_coef'] = model.coef_[0]
            else:
                # For non-linear kernels, coefficients aren't directly
                # interpretable. We'll handle this with permutation
                # importance in the analysis notebook.
                print(f"  SVM using {model.kernel} kernel — "
                      f"no direct feature coefficients available.")
                print(f"  Use permutation importance instead (see notebook).")

        elif name == 'Random Forest':
            importance_data['RF_importance'] = model.feature_importances_

    importance_df = pd.DataFrame(importance_data)

    # Sort by Random Forest importance (usually the most reliable)
    if 'RF_importance' in importance_df.columns:
        importance_df = importance_df.sort_values(
            'RF_importance', ascending=False
        ).reset_index(drop=True)

    print("\nTop 15 most important k-mers (by Random Forest):")
    if 'RF_importance' in importance_df.columns:
        top15 = importance_df.head(15)
        for _, row in top15.iterrows():
            rf_val = f"{row['RF_importance']:.4f}"
            lr_val = f"{row.get('LR_coef', 'N/A'):>+.4f}" if 'LR_coef' in row else "N/A"
            print(f"  {row['kmer']:<6}  RF: {rf_val}  LR: {lr_val}")

    return importance_df


# =============================================================================
# SECTION 8: SAVING RESULTS
# =============================================================================

def save_results(results, comparison_df, importance_df,
                 output_dir='results/metrics'):
    """
    Save all results to CSV files for later analysis and paper figures.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Save comparison table
    comparison_df.to_csv(
        os.path.join(output_dir, 'model_comparison.csv'), index=False
    )

    # Save feature importance
    importance_df.to_csv(
        os.path.join(output_dir, 'feature_importance.csv'), index=False
    )

    # Save detailed metrics for each model
    for name, result in results.items():
        safe_name = name.lower().replace(' ', '_')
        metrics = {k: v for k, v in result['metrics'].items()
                   if not isinstance(v, np.ndarray)}  # Skip arrays (fpr, tpr)
        pd.DataFrame([metrics]).to_csv(
            os.path.join(output_dir, f'{safe_name}_metrics.csv'), index=False
        )

    print(f"\nResults saved to {output_dir}/")
