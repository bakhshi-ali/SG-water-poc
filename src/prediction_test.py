import os
import pickle
import sys

import joblib
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import seaborn as sns
import shap
import xgboost as xgb
import yaml
from catboost import CatBoostClassifier
from imblearn.over_sampling import (ADASYN, SMOTE, SMOTEN, SMOTENC, SVMSMOTE,
                                    BorderlineSMOTE, KMeansSMOTE,
                                    RandomOverSampler)
from IPython.display import display
from joblib import dump, load
from matplotlib import pyplot as plt
from plotly.offline import iplot
from pyecharts import options as opts
from pyecharts.charts import Gauge
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report,
                             cohen_kappa_score, confusion_matrix, f1_score,
                             log_loss, make_scorer, matthews_corrcoef,
                             mean_absolute_error, mean_squared_error,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC, OneClassSVM
from sklearn.utils import shuffle
from xgboost import DMatrix, XGBClassifier

# Get the absolute path of the current script
current_script_path = os.path.abspath(__file__)

# Get the parent directory path
parent_dir_path = os.path.dirname(os.path.dirname(current_script_path))

# Load config file
with open(os.path.dirname(current_script_path) + "/config.yml", 'r') as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.FullLoader)

# Select model
models = config['models']
model = config['model']
data_types = config['data_types']
data_type = config['data_type']
do_tuning = config['do_tuning']
do_over_sampling = config['do_over_sampling']
over_sampling_methods = config['over_sampling_methods']
over_sampling_method = config['over_sampling_method']
n_week_to_predict = config['n_week_to_predict']
n_days_to_predict = config['n_days_to_predict']
n_cv_fold = config['n_cv_fold']
pump_id = config['pump_id']
_suffix = config['suffix']
lagged_days = config['lagged_days']
threshold = config['threshold']

plots_save_path = os.path.join(parent_dir_path, "plots")

# aggregated_columns_{lags}_days' or merged_columns_{lagged_days}_days
lag_features = f'{data_type}_columns_{lagged_days}_days'
# Dataset path
data_dir = os.path.join(parent_dir_path, 'pre-processed-data',
                        f'{data_type}_lagged_features_{lagged_days}_days.csv')   # merged_lagged_features_{lagged_days}_days
# Load the dataset
data = pd.read_csv(data_dir,
                   parse_dates=['DateTime'], index_col='DateTime', dtype=float)

data = data.dropna()

# Create a dataframe to store predicted probability of faults
df_pred = data.copy()


def normalize_input(X):
    # Replace -1 values with NaN
    X[X == -1] = np.nan

    # Create a scaler object
    scaler = MinMaxScaler(feature_range=(0, 2))

    # Fit the scaler on the data and transform the feature values
    X_norm = scaler.fit_transform(X)

    # Replace NaN values with -1 again
    X_norm[np.isnan(X_norm)] = -1

    # Convert the numpy array to a dataframe
    X_norm_df = pd.DataFrame(X_norm, index=X.index, columns=X.columns)

    return X_norm_df


def prepare_data(data):
    _target_cols = []
    # Select and remove columns
    feature_cols = data.columns.tolist()
    for _pump in range(1, 4):
        for _day in range(1, 8):
            feature_cols.remove(
                f'KTF{_pump}_Fault_Status_next_{_day}_day(s)_sum')
            feature_cols.remove(
                f'KTF{_pump}_Fault_Status_next_{_day}_day(s)')
            _target_cols.append(
                f'KTF{_pump}_Fault_Status_next_{_day}_day(s)_sum')
            _target_cols.append(
                f'KTF{_pump}_Fault_Status_next_{_day}_day(s)')

    # Separate the features and the target variable
    X = data.drop(columns=_target_cols)
    # X = X.dropna()

    y1 = data[f'KTF1_Fault_Status_next_{n_days_to_predict}_day(s){_suffix}']
    y1 = y1.loc[X.index]
    y1 = y1.where(y1 == 0, 1)
    # y1 = (y1 > threshold).astype(int)
    # y1.value_counts()
    # y1.value_counts().plot.pie(autopct="%0.2f")
    y2 = data[f'KTF2_Fault_Status_next_{n_days_to_predict}_day(s){_suffix}']
    y2 = y2.loc[X.index]
    y2 = y2.where(y2 == 0, 1)
    # y2 = (y2 > threshold).astype(int)
    # plt.figure()
    # y2.value_counts()
    # y2.value_counts().plot.pie(autopct="%0.2f")
    y3 = data[f'KTF3_Fault_Status_next_{n_days_to_predict}_day(s){_suffix}']
    y3 = y3.loc[X.index]
    y3 = y3.where(y3 == 0, 1)
    # y3 = (y3 > threshold).astype(int)
    # plt.figure()
    # y3.value_counts()
    # y3.value_counts().plot.pie(autopct="%0.2f")

    return X, y1, y2, y3


X, y1, y2, y3 = prepare_data(data)
X = normalize_input(X)
# Split the data into training and testing sets to ensure avoiding data leakage
# Specify the cutoff date for train/test split
cutoff_date = data.index[int(len(data) * 0.8)]
# Split the data into training and testing sets based on cutoff date
X_train = X[X.index <= cutoff_date]
y1_train = y1[y1.index <= cutoff_date]
y2_train = y2[y2.index <= cutoff_date]
y3_train = y3[y3.index <= cutoff_date]
# Shuffle the train set without introducing data leakage
# X_train, y1_train, y2_train, y3_train = shuffle(
#     X_train, y1_train, y2_train, y3_train, random_state=42)
# Training set class distribution
print('Class distribution in y1_train:', y1_train.value_counts())
print('Class distribution in y2_train:', y2_train.value_counts())
print('Class distribution in y3_train:', y3_train.value_counts())
X_test = X[X.index > cutoff_date]
y1_test = y1[y1.index > cutoff_date]
y2_test = y2[y2.index > cutoff_date]
y3_test = y3[y3.index > cutoff_date]
# X_test, y1_test, y2_test, y3_test = shuffle(
#     X_test, y1_test, y2_test, y3_test, random_state=42)
# Test set class distribution
print('Class distribution in y1_test:', y1_test.value_counts())
print('Class distribution in y2_test:', y2_test.value_counts())
print('Class distribution in y3_test:', y3_test.value_counts())


# Train a Random Forest classifier for each pump separately
def random_forest(X_train, y1_train, y2_train, y3_train):
    X_train1, X_train2, X_train3 = X_train[0], X_train[1], X_train[2]
    rf1 = RandomForestClassifier(
        class_weight='balanced', n_estimators=100, random_state=42)
    rf1.fit(X_train1, y1_train)

    rf2 = RandomForestClassifier(n_estimators=100, random_state=42)
    rf2.fit(X_train2, y2_train)

    rf3 = RandomForestClassifier(n_estimators=100, random_state=42)
    rf3.fit(X_train3, y3_train)
    return rf1, rf2, rf3

# Train a Logistic Regression classifier for each pump separately


def logistic_regression(X_train, y1_train, y2_train, y3_train):
    X_train1, X_train2, X_train3 = X_train[0], X_train[1], X_train[2]
    lreg1 = LogisticRegression(max_iter=100000)
    lreg1.fit(X_train1, y1_train)

    lreg2 = LogisticRegression(max_iter=100000)
    lreg2.fit(X_train2, y2_train)

    lreg3 = LogisticRegression(max_iter=100000)
    lreg3.fit(X_train3, y3_train)
    return lreg1, lreg2, lreg3

# Train a Naive Bayes classifier for each pump separately


def naive_bayes(X_train, y1_train, y2_train, y3_train):
    X_train1, X_train2, X_train3 = X_train[0], X_train[1], X_train[2]
    naive1 = GaussianNB()
    naive1.fit(X_train1, y1_train)

    naive2 = GaussianNB()
    naive2.fit(X_train2, y2_train)

    naive3 = GaussianNB()
    naive3.fit(X_train3, y3_train)
    return naive1, naive2, naive3

# Train a Support Vector Machine classifier for each pump separately


def svm(X_train, y1_train, y2_train, y3_train):
    X_train1, X_train2, X_train3 = X_train[0], X_train[1], X_train[2]
    svc1 = SVC(probability=True)
    svc1.fit(X_train1, y1_train)

    svc2 = SVC(probability=True)
    svc2.fit(X_train2, y2_train)

    svc3 = SVC(probability=True)
    svc3.fit(X_train3, y3_train)
    return svc1, svc2, svc3


def svm_one_class(X_train, y1_train, y2_train, y3_train):
    X_train1, X_train2, X_train3 = X_train[0], X_train[1], X_train[2]
    # Define the One-Class SVM models
    ocsvm1 = OneClassSVM(kernel='rbf', nu=0.01, gamma='auto')
    ocsvm2 = OneClassSVM(kernel='rbf', nu=0.01, gamma='auto')
    ocsvm3 = OneClassSVM(kernel='rbf', nu=0.01, gamma='auto')

    # Train the models on the training data for each Pump separately
    ocsvm1.fit(X_train1[y1_train == 0])
    ocsvm2.fit(X_train2[y2_train == 0])
    ocsvm3.fit(X_train3[y3_train == 0])

    # Predict on the testing data for each Pump separately
    y1_pred = ocsvm1.predict(X_test)
    y2_pred = ocsvm2.predict(X_test)
    y3_pred = ocsvm3.predict(X_test)

    return ocsvm1, ocsvm2, ocsvm3


# Train and fit a Gradient Boosting model for each label
def gradient_boosting(X_train, y1_train, y2_train, y3_train):
    X_train1, X_train2, X_train3 = X_train[0], X_train[1], X_train[2]
    gb1 = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb1.fit(X_train1, y1_train)

    gb2 = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb2.fit(X_train2, y2_train)

    gb3 = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb3.fit(X_train3, y3_train)
    return gb1, gb2, gb3


def catboost(X_train, y1_train, y2_train, y3_train):
    X_train1, X_train2, X_train3 = X_train[0], X_train[1], X_train[2]

    # Create a CatBoostClassifier object
    cat1 = CatBoostClassifier(
        n_estimators=200, random_seed=42000, class_weights=[10, 1])

    # Fit the classifier on the data
    cat1.fit(X_train1, y1_train)

    cat2 = CatBoostClassifier(
        n_estimators=200, random_seed=42000, class_weights=[10, 1])
    cat2.fit(X_train2, y2_train)

    cat3 = CatBoostClassifier(
        n_estimators=200, random_seed=42000, class_weights=[10, 1])
    cat3.fit(X_train3, y3_train)

    return cat1, cat2, cat3


def adaboost(X_train, y1_train, y2_train, y3_train):
    X_train1, X_train2, X_train3 = X_train[0], X_train[1], X_train[2]

    # Create an AdaBoostClassifier object
    ada1 = AdaBoostClassifier(n_estimators=200, random_state=42000)

    # Fit the classifier on the data
    ada1.fit(X_train1, y1_train)

    ada2 = AdaBoostClassifier(n_estimators=200, random_state=42)
    ada2.fit(X_train2, y2_train)

    ada3 = AdaBoostClassifier(n_estimators=200, random_state=42)
    ada3.fit(X_train3, y3_train)

    return ada1, ada2, ada3

# import lightgbm as lgb

# def lightgbm(X_train, y1_train, y2_train, y3_train):
#     X_train1, X_train2, X_train3 = X_train[0], X_train[1], X_train[2]

#     # Create a LightGBM Classifier object
#     lgb1 = lgb.LGBMClassifier(n_estimators=200, random_seed=42000, class_weights=[10, 1])

#     # Fit the classifier on the data
#     lgb1.fit(X_train1, y1_train)

#     lgb2 = lgb.LGBMClassifier(n_estimators=200, random_seed=42000, class_weights=[10, 1])
#     lgb2.fit(X_train2, y2_train)

#     lgb3 = lgb.LGBMClassifier(n_estimators=200, random_seed=42000, class_weights=[10, 1])
#     lgb3.fit(X_train3, y3_train)

#     return lgb1, lgb2, lgb3


# Train and fit a Neural Network model for each label
def neural_network(X_train, y1_train, y2_train, y3_train):
    X_train1, X_train2, X_train3 = X_train[0], X_train[1], X_train[2]
    # Scale the input data
    scaler = StandardScaler()
    X_train1 = scaler.fit_transform(X_train1)
    X_train2 = scaler.fit_transform(X_train2)
    X_train3 = scaler.fit_transform(X_train3)

    # Train and fit a neural network model for each label
    class_weights = {0: 0.6, 1: 0.4}
    nn1 = MLPClassifier(hidden_layer_sizes=(50, 25, 10),
                        max_iter=10000,
                        random_state=42,
                        )
    nn1.fit(X_train1, y1_train)

    nn2 = MLPClassifier(hidden_layer_sizes=(50, 25, 10),
                        max_iter=10000,
                        random_state=42,
                        )
    nn2.fit(X_train2, y2_train)

    nn3 = MLPClassifier(hidden_layer_sizes=(50, 25, 10),
                        max_iter=10000,
                        random_state=42,
                        )
    nn3.fit(X_train3, y3_train)
    return nn1, nn2, nn3


def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """Focal loss for binary classification"""
    # Convert probabilities to logits
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    logits = np.log(y_pred / (1 - y_pred))
    # Compute the focal loss
    loss = alpha * (1 - y_pred) ** gamma * y_true * logits + \
        (1 - alpha) * y_pred ** gamma * (1 - y_true) * logits
    return "focal_loss", np.mean(loss), True


def focal_obj(preds, dtrain):
    labels = dtrain.get_label()
    alpha = 0.25  # class balancing factor
    gamma = 2  # focusing parameter
    preds = 1.0 / (1.0 + np.exp(-preds))  # sigmoid
    pt = preds * labels + (1 - preds) * (1 - labels)
    w = alpha * labels + (1 - alpha) * (1 - labels)
    fl = -w * (1 - pt) ** gamma * np.log(pt)
    grad = -w * gamma * (1 - pt) ** (gamma - 1) * (labels - preds)
    hess = w * gamma * (gamma - 1) * (pt * (1 - pt)) ** (gamma - 2)
    return grad, hess


# Define custom evaluation metric
def f1_eval(preds, dtrain):
    labels = dtrain.get_label()
    # convert probabilities to binary predictions
    preds = (preds > 0.5).astype(int)
    return 'f1', f1_score(labels, preds)


def eval_recall(preds, dtrain):
    labels = dtrain.get_label()
    y_preds = np.round(preds)
    recall = recall_score(labels, y_preds)
    return [('recall', recall)]


# Train a XGBoost classifier for each pump separately
def xgboost(X_train, y1_train, y2_train, y3_train, do_tuning):
    X_train1, X_train2, X_train3 = X_train[0], X_train[1], X_train[2]

    if do_tuning:
        clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        params = {
            'max_depth': [3, 6, 10],
            'learning_rate': [0.001, 0.01, 0.1],
            'n_estimators': [100, 500, 1000],
            'colsample_bytree': [0.3, 0.7]
        }
        clf = GridSearchCV(estimator=clf, param_grid=params,
                           scoring='neg_log_loss', verbose=1)

        # Fit and save the best model for each subset of the training data
        clf1 = clf.fit(X_train1, y1_train)
        print('clf1.best_params_')
        print(clf1.best_params_)
        # params1 = {'colsample_bytree': 0.7, 'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 1000}
        # {'colsample_bytree': 0.3, 'learning_rate': 0.001, 'max_depth': 10, 'n_estimators': 500}  for 7 days
        xgb1 = XGBClassifier(use_label_encoder=False,
                             eval_metric='logloss', **clf1.best_params_)
        xgb1.fit(X_train1, y1_train)

        clf2 = clf.fit(X_train2, y2_train)
        print('clf2.best_params_')
        print(clf2.best_params_)
        # params2 = {'colsample_bytree': 0.3, 'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 1000}
        # {'colsample_bytree': 0.3, 'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 500}   for 7 days
        xgb2 = XGBClassifier(use_label_encoder=False,
                             eval_metric='logloss', **clf2.best_params_)
        xgb2.fit(X_train2, y2_train)

        clf3 = clf.fit(X_train3, y3_train)
        print('clf3.best_params_')
        print(clf3.best_params_)
        # params3 = {'colsample_bytree': 0.3, 'learning_rate': 0.01, 'max_depth': 10, 'n_estimators': 1000}
        # {'colsample_bytree': 0.3, 'learning_rate': 0.001, 'max_depth': 10, 'n_estimators': 1000} for 7 days
        xgb3 = XGBClassifier(use_label_encoder=False,
                             eval_metric='logloss', **clf3.best_params_)
        xgb3.fit(X_train3, y3_train)

    else:

        # # Define the scale_pos_weight parameter
        scale_pos_weight_1 = len(
            y1_train[y1_train == 0]) / len(y1_train[y1_train == 1])
        # Define default model parameters
        xgb1 = XGBClassifier(n_estimators=100, scale_pos_weight=scale_pos_weight_1*0.6,
                             objective='binary:logistic', eval_metric=eval_recall, random_state=42)
        xgb1.fit(X_train1, y1_train)

        scale_pos_weight_2 = len(
            y2_train[y2_train == 0]) / len(y2_train[y2_train == 1])
        xgb2 = XGBClassifier(n_estimators=100, scale_pos_weight=scale_pos_weight_2*0.5,
                             objective='binary:logistic', eval_metric=eval_recall, random_state=42)
        xgb2.fit(X_train2, y2_train)

        scale_pos_weight_3 = len(
            y3_train[y3_train == 0]) / len(y3_train[y3_train == 1])
        xgb3 = XGBClassifier(n_estimators=100, scale_pos_weight=scale_pos_weight_3*0.6,
                             objective='binary:logistic', eval_metric=eval_recall, random_state=42)
        xgb3.fit(X_train3, y3_train)

        # Fit each classifier with default hyperparameters
        # xgb1 = XGBClassifier(use_label_encoder=False,
        #                      objective='rank:pairwise',
        #                      n_estimators=100,
        #                      max_depth=5,
        #                      min_child_weight=1,
        #                      gamma=0,
        #                      subsample=0.8,
        #                      colsample_bytree=0.8,
        #                      reg_alpha=0.005,
        #                      seed=42,
        #                      eval_metric='auc')
        # xgb1.fit(X_train1, y1_train)

        # xgb2 = XGBClassifier(use_label_encoder=False,
        #                      objective='binary:logistic',
        #                      learning_rate=0.1,
        #                      n_estimators=100,
        #                      max_depth=5,
        #                      min_child_weight=1,
        #                      gamma=0,
        #                      subsample=0.8,
        #                      colsample_bytree=0.8,
        #                      reg_alpha=0.005,
        #                      seed=42,
        #                      eval_metric='auc')
        # xgb2.fit(X_train2, y2_train)

        # xgb3 = XGBClassifier(use_label_encoder=False,
        #                      objective='binary:logistic',
        #                      learning_rate=0.1,
        #                      n_estimators=100,
        #                      max_depth=5,
        #                      min_child_weight=1,
        #                      gamma=0,
        #                      subsample=0.8,
        #                      colsample_bytree=0.8,
        #                      reg_alpha=0.005,
        #                      seed=42,
        #                      eval_metric='auc')
        # xgb3.fit(X_train3, y3_train)

    return xgb1, xgb2, xgb3


# Train a XGBoost classifier with DMMatrix
def xgboost_dm(X_train, y1_train, y2_train, y3_train, do_tuning):
    X_train1, X_train2, X_train3 = X_train[0], X_train[1], X_train[2]

    # Define the scale_pos_weight parameter
    scale_pos_weight_1 = len(
        y1_train[y1_train == 0]) / len(y1_train[y1_train == 1])
    scale_pos_weight_2 = len(
        y2_train[y2_train == 0]) / len(y2_train[y2_train == 1])
    scale_pos_weight_3 = len(
        y3_train[y3_train == 0]) / len(y3_train[y3_train == 1])

    # Create the DMatrix object with the specified weights
    dtrain1 = xgb.DMatrix(X_train1, label=y1_train, weight=[
        scale_pos_weight_1 if i == 1 else 1 for i in y1_train])
    dtrain2 = xgb.DMatrix(X_train2, label=y2_train, weight=[
        scale_pos_weight_2 if i == 1 else 1 for i in y2_train])
    dtrain3 = xgb.DMatrix(X_train3, label=y3_train, weight=[
        scale_pos_weight_3 if i == 1 else 1 for i in y3_train])

    # Define default model parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'random_state': 42,
        'scale_pos_weight': 0.6
    }

    # Train the XGBoost models using the DMatrix objects
    xgb1 = xgb.train(params, dtrain1, num_boost_round=100)
    xgb2 = xgb.train(params, dtrain2, num_boost_round=100)
    xgb3 = xgb.train(params, dtrain3, num_boost_round=100)

    return xgb1, xgb2, xgb3


# Oversampling techniques like SVMSMOTE are used to address class imbalance in a dataset. Class imbalance occurs
# When one class has significantly fewer samples than the other(s), which can lead to biased models that perform poorly on the minority class.
def over_sampling(X_train, y1_train, y2_train, y3_train, over_sampling_method):
    # The key idea behind SVM-SMOTE is to use the SVM algorithm to identify the most important minority class samples, and to generate synthetic
    # samples that are more likely to be correctly classified by the SVM. This can lead to better performance compared to other oversampling methods
    # that do not take into account the classifier being used.
    if over_sampling_method == 'svmsmote':
        method = SVMSMOTE(random_state=42)
    elif over_sampling_method == 'smote':
        # Generates synthetic samples by interpolating between existing samples of the minority class.
        method = SMOTE(random_state=42)
    elif over_sampling_method == 'borderline-smote':
        # A variation of SMOTE that only generates synthetic samples for borderline samples that are close to the decision boundary.
        method = BorderlineSMOTE(random_state=42)
    elif over_sampling_method == 'self-level-smote':
        # A variation of SMOTE that only generates synthetic samples for safe-level samples, which are minority samples
        # that are close to majority samples but not too close.
        method = SMOTEN(random_state=0)
    elif over_sampling_method == 'kmeans-smote':
        # Instead of using nearest neighbors to generate synthetic samples, KMeansSMOTE clusters the minority class samples using
        # k-means clustering and selects the representative samples from each cluster to generate new synthetic samples.
        method = KMeansSMOTE(kmeans_estimator=MiniBatchKMeans(
            n_init=1, random_state=0), random_state=42)
    elif over_sampling_method == 'ADASYN':
        # Similar to SMOTE, but generates more synthetic samples for difficult-to-learn examples by using a density distribution.
        method = ADASYN(random_state=42)
        # Designed to work with datasets that contain both continuous and categorical features.
    elif over_sampling_method == 'smotenc':
        method = SMOTENC(random_state=42, categorical_features=[18, 19])
    else:
        # Randomly duplicate samples from the minority class to increase its representation in the dataset.
        method = RandomOverSampler(random_state=42)
    X_train_smote1, y1_train_smote = method.fit_resample(X_train, y1_train)
    X_train_smote2, y2_train_smote = method.fit_resample(X_train, y2_train)
    X_train_smote3, y3_train_smote = method.fit_resample(X_train, y3_train)
    print(f"{over_sampling_method} resampled data generated.")
    X_train_smote = [X_train_smote1, X_train_smote2, X_train_smote3]
    return X_train_smote, y1_train_smote, y2_train_smote, y3_train_smote


def train_models(X_train, y1_train, y2_train, y3_train, do_tuning, model):
    # models = ['Random Forest', 'XGBoost', 'Logistic Regression', 'Naive Bayes', 'Support Vector Machine', 'Gradient Boosting', 'Neural Network']
    if model == models[0]:
        model1, model2, model3 = random_forest(
            X_train, y1_train, y2_train, y3_train)
    elif model == models[1]:
        model1, model2, model3 = xgboost(
            X_train, y1_train, y2_train, y3_train, do_tuning)
    elif model == models[2]:
        model1, model2, model3 = logistic_regression(
            X_train, y1_train, y2_train, y3_train)
    elif model == models[3]:
        model1, model2, model3 = naive_bayes(
            X_train, y1_train, y2_train, y3_train)
    elif model == models[4]:
        model1, model2, model3 = svm(X_train, y1_train, y2_train, y3_train)
    elif model == models[5]:
        model1, model2, model3 = gradient_boosting(
            X_train, y1_train, y2_train, y3_train)
    elif model == models[6]:
        model1, model2, model3 = neural_network(
            X_train, y1_train, y2_train, y3_train)
    elif model == models[7]:
        model1, model2, model3 = catboost(
            X_train, y1_train, y2_train, y3_train)
    elif model == models[8]:
        model1, model2, model3 = adaboost(
            X_train, y1_train, y2_train, y3_train)
    elif model == model[9]:
        model1, model2, model3 = xgboost_dm(
            X_train, y1_train, y2_train, y3_train)

    return model1, model2, model3


def training_pipline(X_train, y1_train, y2_train, y3_train, over_sampling_method):

    if do_over_sampling:
        X_train, y1_train, y2_train, y3_train = over_sampling(
            X_train, y1_train, y2_train, y3_train, over_sampling_method)
    else:
        X_train = [X_train, X_train, X_train]

    model1, model2, model3 = train_models(
        X_train, y1_train, y2_train, y3_train, do_tuning, model)
    return X_train, y1_train, y2_train, y3_train, model1, model2, model3


def _calculate_metrics(y_true, y_scores, y_pred):
    # y_true: true labels of the data
    # y_scores: predicted scores or probabilities

    # The ROC curve shows the trade-off between true positive rate (the proportion of actual positives
    # that are correctly identified) and false positive rate (the proportion of actual negatives that
    # are incorrectly identified as positives) for different classification threshold values.
    # AUC (Area Under the Curve) measures the overall ability of the classifier to distinguish between
    # positive and negative samples. You can describe this as the classifier's ability to identify true
    # positives while minimizing false positives.
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc_score = roc_auc_score(y_true, y_scores)
    print(f'AUC={auc_score:.2f}')
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC={auc_score:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    # Precision-Recall Curve and AUC Score
    # PR-AUC (Precision-Recall Area Under the Curve) measures the overall ability of the classifier to correctly
    # identify positive samples while minimizing false positives.
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc_score = average_precision_score(y_true, y_scores)
    print(f'PR AUC={pr_auc_score:.2f}')
    plt.figure()
    plt.plot(recall, precision, label=f'PR AUC={pr_auc_score:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()

    # Log Loss measures the performance of a classifier that predicts probabilities or scores for each class. It penalizes
    # the classifier for being confident and wrong, which can be important in situations where the cost of false positives
    # and false negatives are different. You can describe this as a measure of how well the classifier's predicted probabilities
    # match the actual probabilities of the classes.
    logloss = log_loss(y_true, y_pred)
    print(f'Log Loss: {logloss:.4f}')

    # Cohen's Kappa measures the agreement between two raters (in this case, the classifier and the true labels) and is useful
    # for evaluating the performance of a classifier when the dataset is imbalanced. You can describe this as a measure of how
    # much better the classifier performs than a random guess, taking into account the fact that the dataset may have a skewed
    # distribution of classes
    kappa = cohen_kappa_score(y_true, y_pred)
    print(f'Cohen\'s Kappa: {kappa:.4f}')

    # MCC is a measure of the quality of binary (two-class) classifications that takes into account true and false positives and
    # negatives. It can be used to evaluate the performance of a classifier when the dataset is imbalanced. You can describe this
    # as a balanced measure that takes into account both positive and negative predictions.
    mcc = matthews_corrcoef(y_true, y_pred)
    print(f'Matthews Correlation Coefficient: {mcc:.4f}')

    # MAE and MSE are regression metrics used to evaluate the performance of continuous variables. MAE measures the average magnitude
    # of the errors in a set of predictions, while MSE measures the average squared difference between the predicted and actual values.
    # You can describe this as a measure of how close the predicted values are to the actual values, with MAE giving more weight to large
    # errors and MSE giving more weight to very large errors.
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    print(f'Mean Absolute Error: {mae:.4f}')
    print(f'Mean Squared Error: {mse:.4f}')

    return auc_score, pr_auc_score, logloss, kappa, mcc, mae, mse


def evaluate_models(model, model1, model2, model3, X_test, y1_test, y2_test, y3_test):
    # Convert test data to DMatrix
    if model == 'XGBoost-DM':
        X_test = xgb.DMatrix(X_test)
        y1_prob = model1.predict(X_test)
        y2_prob = model2.predict(X_test)
        y3_prob = model3.predict(X_test)
        X1 = xgb.DMatrix(X)
        y1_prob_all = model1.predict(X1)
        y2_prob_all = model2.predict(X1)
        y3_prob_all = model3.predict(X1)
    else:
        # calculate the predicted probabilities of fault for each pump on the test set
        y1_prob = model1.predict_proba(X_test)[:, 1]
        y2_prob = model2.predict_proba(X_test)[:, 1]
        y3_prob = model3.predict_proba(X_test)[:, 1]
        # predict the probabilities of fault for each pump on the whole dataset
        y1_prob_all = model1.predict_proba(X)[:, 1]
        y2_prob_all = model2.predict_proba(X)[:, 1]
        y3_prob_all = model3.predict_proba(X)[:, 1]

    # calculate the AUC scores for each pump
    auc1 = roc_auc_score(y1_test, y1_prob)
    auc2 = roc_auc_score(y2_test, y2_prob)
    auc3 = roc_auc_score(y3_test, y3_prob)

    # Call save_output_to_file function and pass in the file name you want to use
    # save_output_to_file(os.path.join(parent_dir_path, "models-outputs", f"{model}_output.txt"))

    # print the AUC scores
    print(f"AUC for KTF1: {auc1}")
    print(f"AUC for KTF2: {auc2}")
    print(f"AUC for KTF3: {auc3}")

    # convert the predicted probabilities into binary labels using a threshold of 0.5
    y1_pred = np.where(y1_prob >= 0.5, 1, 0)
    y2_pred = np.where(y2_prob >= 0.5, 1, 0)
    y3_pred = np.where(y3_prob >= 0.5, 1, 0)

    # y1_pred = model1.predict(X_test)
    # y2_pred = model2.predict(X_test)
    # y3_pred = model3.predict(X_test)

    """
    A brief description of each metric in classification report
    Precision: The precision metric measures the proportion of true positives (correctly predicted positive instances)
    among all predicted positives. It provides information about the model's ability to avoid false positives. A high
    precision score indicates that the model makes few false positive predictions.
    Recall: The recall metric measures the proportion of true positives among all actual positives. It provides information
    about the model's ability to identify positive instances. A high recall score indicates that the model identifies most positive instances.
    F1-score: The F1-score is a harmonic mean of precision and recall. It is a way to combine precision and recall metrics into a single value.
    The F1-score provides a balance between precision and recall. A high F1-score indicates that the model has high precision and high recall.
    Support: The support metric is the number of instances of each class in the true data. It shows how many instances are in each class.
    Accuracy: The accuracy metric measures the proportion of correctly predicted instances among all instances. It provides information about
    the model's overall performance. However, it can be misleading when the classes are imbalanced.
    Macro avg: The macro avg is the average of precision, recall, and F1-score for each class. It provides information about the overall performance
    of the model across all classes. It is calculated by averaging the scores for each class without considering the class imbalance.
    Weighted avg: The weighted avg is the weighted average of precision, recall, and F1-score for each class, weighted by the number of instances
    of each class. It provides information about the overall performance of the model, taking into account the class imbalance.
    """
    # print classification report and AUC score
    print('Classification report for Pump1')
    print(classification_report(y1_test, y1_pred))
    print('Classification report for Pump2')
    print(classification_report(y2_test, y2_pred))
    print('Classification report for Pump3')
    print(classification_report(y3_test, y3_pred))

    # plot confusion matrix
    plt.figure()
    plot_confusion_matrix(
        y1_test, y1_pred, 'Confusion Matrix for KTF1 Fault Prediction on Test Data')
    plt.figure()
    plot_confusion_matrix(
        y2_test, y1_pred, 'Confusion Matrix for KTF2 Fault Prediction on Test Data')
    plt.figure()
    plot_confusion_matrix(
        y3_test, y1_pred, 'Confusion Matrix for KTF3 Fault Prediction on Test Data')

    """
    the moving-average predicted probabilities are the rolling mean (i.e., moving average) of the predicted probabilities
    of each pump's fault status (i.e., probability of fault). This moving average is calculated with a rolling window of
    14 days (24 hours * 7 days * 2 weeks) on the entire dataset. The resulting values represent a smoothed version of the
    predicted probabilities, allowing for better visualization of the trends and patterns over time. The calculated
    moving-average predicted probabilities are then used to calculate the risk scores.
    """

    # calculate the moving-average predicted probabilities with a rolling window of 24 hours * 6 * 7 days * number of week(s)
    # Since data recorded evry 10 minutes, we should multiply the window by 6
    moving_average_y1 = pd.Series(y1_prob_all).rolling(
        window=24*6*n_days_to_predict).mean().fillna(0)
    moving_average_y2 = pd.Series(y2_prob_all).rolling(
        window=24*6*n_days_to_predict).mean().fillna(0)
    moving_average_y3 = pd.Series(y3_prob_all).rolling(
        window=24*6*n_days_to_predict).mean().fillna(0)

    # calculate the risk scores as (moving-average predicted probabilities) * 100
    # Score of 0 or 100 means pump fault (of any severity) is extremly unlikely or likely to occur in the next X days, respectively.
    risk_scores_y1 = moving_average_y1 * 100
    risk_scores_y2 = moving_average_y2 * 100
    risk_scores_y3 = moving_average_y3 * 100

    # print the risk scores
    # risk scores used to evaluate the likelihood of a particular outcome or event based on historical data and other relevant factors
    print("Risk scores for KTF1:")
    print(risk_scores_y1)
    print("Risk scores for KTF2:")
    print(risk_scores_y2)
    print("Risk scores for KTF3:")
    print(risk_scores_y3)

    # calculate the average risk scores
    avg_risk_y1 = risk_scores_y1.mean()
    avg_risk_y2 = risk_scores_y2.mean()
    avg_risk_y3 = risk_scores_y3.mean()

    # print the average risk scores
    print("Average risk score for KTF1:", avg_risk_y1)
    print("Average risk score for KTF2:", avg_risk_y2)
    print("Average risk score for KTF3:", avg_risk_y3)

    plot_guage_chart(avg_risk_y1, 1)
    plot_guage_chart(avg_risk_y2, 2)
    plot_guage_chart(avg_risk_y3, 3)

    # gauge = gauge_plot(avg_risk_y1, 100, 1)
    # show(gauge)

    # Restore output back to console
    # restore_output_to_console()

    add_predicted_probabilities(
        y1_prob_all, moving_average_y1, risk_scores_y1, 1, n_days_to_predict)
    add_predicted_probabilities(
        y2_prob_all, moving_average_y2, risk_scores_y2, 2, n_days_to_predict)
    add_predicted_probabilities(
        y2_prob_all, moving_average_y3, risk_scores_y3, 3, n_days_to_predict)

    auc_score, pr_auc_score, logloss, kappa, mcc, mae, mse = _calculate_metrics(
        y1_test, y1_prob, y1_pred)

    return y1_prob_all, y2_prob_all, y3_prob_all


def plot_guage_chart_html(avg_risk_score, pump_n):
    # Set up the gauge chart
    gauge = Gauge(init_opts=opts.InitOpts(width="500px", height="300px"))
    gauge.add("", [
              (f"Risk Score Pump {pump_n} fault in next {n_days_to_predict} day(s)", avg_risk_score)], min_=0, max_=100)

    # Customize the gauge chart
    gauge.set_global_opts(title_opts=opts.TitleOpts(title="Risk Score"),
                          legend_opts=opts.LegendOpts(is_show=False),
                          tooltip_opts=opts.TooltipOpts(formatter="{a} <br/>{b} : {c}"))

    save_path = os.path.join(parent_dir_path, "plots")

    # Render the gauge chart
    gauge.render(
        save_path + f"/Risk Score Pump {pump_n} fault in next {n_days_to_predict} day(s).html")


def plot_guage_chart(avg_risk_score, pump_n):
    # Create the gauge plot
    gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=avg_risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Risk Score Pump {pump_n} fault in next {n_days_to_predict} day(s)", 'font': {
            'size': 24}, 'align': 'right'
            },
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': 'lightblue'},
                {'range': [50, 70], 'color': 'yellow'},
                {'range': [70, 100], 'color': 'red'}],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': avg_risk_score}}))

    # Show the plot
    display(iplot(gauge))
    # Save plot
    gauge.write_image(plots_save_path + f"/Risk_Score_Pump {pump_n}_fault_in_next_{n_days_to_predict}_day(s).png")



def add_predicted_probabilities(Y_pred, moving_average, risk_score, pump_id, n_days_to_predict):
    label_col = f'pump_{pump_id}_faults_{n_days_to_predict}day(s)'
    df_pred[f'prob_{label_col}'] = Y_pred
    df_pred[f'prob_{label_col}_{7*n_days_to_predict}day_moving_avg'] = moving_average
    df_pred[f'risk_score_pump{pump_id}'] = risk_score

    return df_pred


def cross_validation(model, X_train, X_test, model1, model2, model3, y1_train, y2_train, y3_train, n_cv_fold):
    y1_test_pred = model1.predict(X_test)
    y1_test_pred_proba = model1.predict_proba(X_test)[:, 1]
    y2_test_pred = model2.predict(X_test)
    y2_test_pred_proba = model2.predict_proba(X_test)[:, 1]
    y3_test_pred = model3.predict(X_test)
    y3_test_pred_proba = model3.predict_proba(X_test)[:, 1]

    precision = make_scorer(precision_score, zero_division=0)

    precision1 = cross_val_score(
        model1, X_train[0], y1_train, cv=n_cv_fold, scoring=precision)
    precision2 = cross_val_score(
        model2, X_train[1], y2_train, cv=n_cv_fold, scoring=precision)
    precision3 = cross_val_score(
        model3, X_train[2], y3_train, cv=n_cv_fold, scoring=precision)

    recall_1 = cross_val_score(
        model1, X_train[0], y1_train, cv=n_cv_fold, scoring='recall')
    f1_1 = cross_val_score(model1, X_train[0], y1_train,
                           cv=n_cv_fold, scoring='f1')
    auc_1 = cross_val_score(model1, X_train[0], y1_train,
                            cv=n_cv_fold, scoring='roc_auc')
    recall_2 = cross_val_score(
        model2, X_train[1], y2_train, cv=n_cv_fold, scoring='recall')
    f1_2 = cross_val_score(model2, X_train[1], y2_train,
                           cv=n_cv_fold, scoring='f1')
    auc_2 = cross_val_score(model2, X_train[1], y2_train,
                            cv=n_cv_fold, scoring='roc_auc')
    recall_3 = cross_val_score(
        model3, X_train[2], y3_train, cv=n_cv_fold, scoring='recall')
    f1_3 = cross_val_score(model3, X_train[2], y3_train,
                           cv=n_cv_fold, scoring='f1')
    auc_3 = cross_val_score(model3, X_train[2], y3_train,
                            cv=n_cv_fold, scoring='roc_auc')

    # Save trained models
    save_path = os.path.join(parent_dir_path, "saved-models")
    dump(model1, save_path + f'/model1_{n_days_to_predict}_day(s).joblib')
    dump(model2, save_path + f'/model2_{n_days_to_predict}_day(s).joblib')
    dump(model3, save_path + f'/model3_{n_days_to_predict}_day(s).joblib')

    # Load the models
    # loaded_model1 = joblib.load(save_path + '/model1_{n_days_to_predict}_day(s).joblib')
    # loaded_model2 = joblib.load(save_path + '/model2_{n_days_to_predict}_day(s).joblib')
    # loaded_model3 = joblib.load(save_path + '/model3_{n_days_to_predict}_day(s).joblib')

    # Call save_output_to_file function and pass in the file name you want to use
    # save_output_to_file(os.path.join(parent_dir_path, "models-outputs", f"{model}_output_cv.txt"))

    print("----------------------------------------")
    print(
        f"PREDICTING IF THERE WILL BE ANY PUMPs FAULT IN THE NEXT {n_days_to_predict} DAY(S)")
    print(f"Classifier: {model}")
    print(f" Features: ")
    print(f"    - number of features: {X.shape[1]}")
    print(
        f"    - number of days in the past as features: {n_days_to_predict}")
    print(f"    - Smote oversampling: {do_over_sampling}")
    print(f"    - Hyperparameter tunning: {do_tuning}")
    print(" Data Summary: ")
    print(f"    - Original train data size: {len(y1_train)}")
    print(f"    - Whole data size: {len(y1)}")
    print(f"    - Test data size: {len(y1_test_pred)}")
    print_results(precision1, recall_1, f1_1, auc_1, 1)
    print_results(precision2, recall_2, f1_2, auc_2, 2)
    print_results(precision3, recall_3, f1_3, auc_3, 3)
    print("----------------------------------------")

    # Restore output back to console
    # restore_output_to_console()


def print_results(precision, recall, f1, auc, pump_n):
    print(f"-------- Performance metrics of pump {pump_n} --------")
    print(
        f" -- > Precision (mean of {n_cv_fold}-fold cv): {round(precision.mean(), 3)}")
    print(
        f" -- > Recall (mean of {n_cv_fold}-fold cv): {round(recall.mean(), 3)}")
    print(f" -- > F1 (mean of {n_cv_fold}-fold cv): {round(f1.mean(), 3)}")
    print(
        f" -- > Accuracy (mean of {n_cv_fold}-fold cv): ",  round(auc.mean(), 3))


# Define a function to redirect output to a file
def save_output_to_file(file_name):
    sys.stdout = open(file_name, 'w')

# Define a function to restore the output back to the console


def restore_output_to_console():
    sys.stdout.close()
    sys.stdout = sys.__stdout__


def plot_confusion_matrix(y_true, y_pred, title):
    labels = ['Normal', 'Fault']
    cm = confusion_matrix(y_true, y_pred)
    # plt.figure()
    ax = sns.heatmap(cm, annot=True, cmap='Blues', fmt='g',
                     xticklabels=labels, yticklabels=labels)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title(title)
    plt.show()


def plot_prediction_vs_true_values(y1, y1_prob_all, y2, y2_prob_all, y3, y3_prob_all):
    # Plot model predictions vs true values for the entire period
    fig, axs = plt.subplots(nrows=3, figsize=(10, 10))
    sns.lineplot(ax=axs[0], x=y1.index, y=y1_prob_all,
                 color='deeppink', label='Predicted')
    sns.lineplot(ax=axs[0], x=y1.index, y=y1.values,
                 color='teal', label='True')
    axs[0].set_title('KTF1_Fault_Status')
    axs[0].legend()
    sns.lineplot(ax=axs[1], x=y2.index, y=y2_prob_all,
                 color='deeppink', label='Predicted')
    sns.lineplot(ax=axs[1], x=y2.index, y=y2.values,
                 color='teal', label='True')
    axs[1].set_title('KTF2_Fault_Status')
    axs[1].legend()
    sns.lineplot(ax=axs[2], x=y3.index, y=y3_prob_all,
                 color='deeppink', label='Predicted')
    sns.lineplot(ax=axs[2], x=y3.index, y=y3.values,
                 color='teal', label='True')
    axs[2].set_title('KTF3_Fault_Status')
    axs[2].legend()
    plt.tight_layout()
    plt.show()

    # Plot model predictions vs true values for a specific time interval
    start_date = pd.to_datetime('2022-09-01')
    end_date = pd.to_datetime('2023-02-01')
    # create a boolean mask for the date range
    mask = (data.index >= start_date) & (data.index <= end_date)

    # slice the predicted probabilities using the boolean mask
    y1_prob_slice = y1_prob_all[mask]
    y2_prob_slice = y2_prob_all[mask]
    y3_prob_slice = y3_prob_all[mask]

    fig, axs = plt.subplots(nrows=3, figsize=(10, 10))
    sns.lineplot(ax=axs[0], x=y1[mask].index, y=y1_prob_slice,
                 color='deeppink', label='Predicted')
    sns.lineplot(ax=axs[0], x=y1.loc[start_date:end_date].index,
                 y=y1.loc[start_date:end_date], color='teal', label='True')
    axs[0].set_title(f'KTF1_Fault_Status between {start_date} and {end_date}')
    axs[0].legend()
    sns.lineplot(ax=axs[1], x=y2[mask].index, y=y2_prob_slice,
                 color='deeppink', label='Predicted')
    sns.lineplot(ax=axs[1], x=y2.loc[start_date:end_date].index,
                 y=y2.loc[start_date:end_date], color='teal', label='True')
    axs[1].set_title(f'KTF2_Fault_Status between {start_date} and {end_date}')
    axs[1].legend()
    sns.lineplot(ax=axs[2], x=y3[mask].index, y=y3_prob_slice,
                 color='deeppink', label='Predicted')
    sns.lineplot(ax=axs[2], x=y3.loc[start_date:end_date].index,
                 y=y3.loc[start_date:end_date], color='teal', label='True')
    axs[2].set_title(f'KTF3_Fault_Status between {start_date} and {end_date}')
    axs[2].legend()
    plt.tight_layout()
    plt.show()


def plot_prediction_vs_true(yx, pump_id, n_days_to_predict):
    # Plot model prediction vs. true labels
    label_col = f'pump_{pump_id}_faults_{n_days_to_predict}day(s)'
    df_plt1 = df_pred[[f'KTF{pump_id}_Fault_Status',
                       f'prob_{label_col}',
                      f'prob_{label_col}_{7*n_days_to_predict}day_moving_avg',
                       f'risk_score_pump{pump_id}']]
    df_plt1 = df_plt1.rename(columns={f'KTF{pump_id}_Fault_Status': 'Actual',
                                      f'prob_{label_col}': 'Predicted_prob',
                                      f'prob_{label_col}_{7*n_days_to_predict}day_moving_avg': f'Predicted_prob_{n_days_to_predict}day(s)_avg',
                                      f'risk_score_pump{pump_id}': 'Risk_score'})
    # df_plt1.reset_index(inplace=True)

    # df_plt1 = pd.melt(df_plt1, id_vars='new_index_col_name', value_vars=['Actual', 'Risk_score'])
    # df_plt1['Risk_score'].fillna(df_plt1['Risk_score'].mean(), inplace=True)

    fig, axs = plt.subplots(nrows=1, figsize=(10, 10))
    sns.lineplot(ax=axs, x=yx.index, y='Predicted_prob',
                 data=df_plt1, color='teal', label='True')
    sns.lineplot(ax=axs, x=yx.index, y='Actual', data=df_plt1,
                 color='deeppink', label='Predicted')
    axs.set_title(f'KTF{pump_id}_Fault_Status between ')
    axs.legend()
    axs.set(xlabel="DateTime", ylabel=f"Values",
            title=f"Actual vs. predicted pump{pump_id} fault in the next {n_days_to_predict} day(s)")
    plt.show()


# Compute the SHAP values for the linear model
def clf_log_odds(clf, x):
    p = clf.predict_log_proba(x)
    return p[:, 1] - p[:, 0]


# Calculate log odds
def calculate_log_odds(model, clf, x_train, X_train):
    # clf = RandomForestClassifier(n_estimators=100, random_state=42)
    # clf.fit(x_train, y1_train)
    if model == 'Logistic Regression':
        background = shap.maskers.Independent(X_train, max_samples=100)
        explainer_log_odds = shap.Explainer(
            clf_log_odds(clf, x_train), background)
        shap_values = explainer_log_odds(X_train)
    elif model == 'Naive Bayes':
        log_odds = 0
    elif model == 'Support Vector Machine':
        log_odds = 0
    elif model == 'XGBoost' or model == 'XGBoost-DM':
        background = shap.maskers.Independent(x_train, max_samples=100)
        explainer = shap.Explainer(clf, background)
        shap_values = explainer(x_train)
    elif model == 'Random Forest':
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(x_train)
    elif model == 'Gradient Boosting':
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_train)
    elif model == 'Neural Network':
        explainer = shap.DeepExplainer(clf, X_train)
        shap_values = explainer.shap_values(X_train)
    elif model == 'CatBoost':
        background = shap.maskers.Independent(x_train, max_samples=100)
        explainer = shap.Explainer(clf)
        shap_values = explainer(x_train)
    elif model == 'Adaboost':
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_train)

    else:
        clf = None
        shap_values = None

    # compute the feature names and instances
    feature_names = list(x_train.columns)
    instances = shap.sample(x_train, 100)

    # compute the predicted probabilities and colors
    if model == 'Logistic Regression' or model == 'XGBoost' or model == 'XGBoost-DM':
        probs = clf.predict_proba(instances)[:, 1]
    else:
        probs = clf.predict_proba(x_train)[:, 1]
    color_map = cm.get_cmap('coolwarm')
    colors = color_map(probs)

    # combine the SHAP values if they are returned as a list
    if isinstance(shap_values, list):
        shap_values = np.sum(shap_values, axis=0)

    # wrap the SHAP values in an Explanation object
    if isinstance(shap_values, np.ndarray):
        shap_values = shap.Explanation(
            shap_values, x_train, feature_names=feature_names)

    return shap_values, feature_names, colors


def compute_shape_value(model, model1, x_train, X_train):

    shap_values, feature_names, colors = calculate_log_odds(
        model, model1, x_train, X_train)
    # pass the colors array to the summary_plot function
    shap.summary_plot(shap_values, max_display=30,
                      feature_names=feature_names, color=colors)
    # create new plot figure for bar plot
    plt.figure()
    # shap.plots.beeswarm(shap_values)
    shap.plots.bar(shap_values, show_data='auto', show=False)
    # create new plot figure for bar plot
    plt.figure()
    shap.plots.bar(shap_values.abs.max(0))
    # create new plot figure for bar plot
    # plt.figure()
    # shap.plots.heatmap(shap_values)
    plt.show()


def global_shap_importance(model, model1, X):
    """ Return a dataframe containing the features sorted by Shap importance
    Parameters
    ----------
    model : The tree-based model
    X : pd.Dataframe
         training set/test set/the whole dataset ... (without the label)
    Returns
    -------
    pd.Dataframe
        A dataframe containing the features sorted by Shap importance
    """
    # Create a SHAP explainer object for the provided model.
    # explainer = shap.Explainer(model)
    # Use the explainer to generate the SHAP values for the provided dataset X
    # shap_values = explainer(X)

    shap_values, feature_names, colors = calculate_log_odds(
        model, model1, X, X)

    # Create a dictionary of SHAP values with a single key-value pair with the
    # key being an empty string and the value being the SHAP values generated in the previous step.
    cohorts = {"": shap_values}
    # Store the key in a list called cohort_labels.
    cohort_labels = list(cohorts.keys())
    # Store the value in a list called cohort_exps
    cohort_exps = list(cohorts.values())
    # Iterate through the cohort_exps list and check if the shape of the SHAP values is two-dimensional.
    for i in range(len(cohort_exps)):
        if len(cohort_exps[i].shape) == 2:
            # Compute the mean absolute value for each feature across all the samples.
            cohort_exps[i] = cohort_exps[i].abs.mean(0)
    # Extract the feature data from the first element of the cohort_exps list and store it in features
    features = cohort_exps[0].data
    # Extract the feature names from the first element of the cohort_exps list and store it in feature_names
    feature_names = cohort_exps[0].feature_names
    # Extract the SHAP values for each feature for each sample and store it in a numpy array called values
    values = np.array([cohort_exps[i].values for i in range(len(cohort_exps))])
    # Create a pandas dataframe feature_importance with columns features and importance, where features contains the names of
    # the features and importance contains the sum of the SHAP values for each feature across all the samples.
    feature_importance = pd.DataFrame({'features': feature_names,
                                       'importance': sum(values)})
    # Sort the dataframe by importance in descending order.
    feature_importance.sort_values(
        by=['importance'], ascending=False, inplace=True)
    return feature_importance


def generate_shap_feature_importance(model, model1):

    df_fh = global_shap_importance(model, model1, X)
    # Load lagged_cols from the file (generated in feature_engineering step)
    with open(os.path.join(parent_dir_path, "pre-processed-data", f"{lag_features}.pkl"), 'rb') as f:
        lagged_columns = pickle.load(f)
    # Create a new dataframe with two column
    df_features = pd.DataFrame(columns=['feature', 'score'])
    for name in lagged_columns:
        # print('name', name)
        df_fh_c = df_fh.copy()
        df_fh_c["group"] = df_fh_c["features"].str.contains(name, regex=False)
        #  The groupby operation creates two groups - one where the feature name contains the lagged column name and one where it doesn't
        df_fh_c = df_fh_c.groupby("group").sum(numeric_only=True)
        score = df_fh_c.loc[True].values[0]
        # df_features = df_features.append({'feature': name, 'score': score}, ignore_index=True)
        df_features = pd.concat([df_features, pd.DataFrame(
            {'feature': [name], 'score': [score]})], ignore_index=True)

    df_features = df_features.sort_values(by=['score'], ascending=False)
    df_features.to_excel(os.path.join(parent_dir_path,
                                      f"pump{pump_id}_fault_clf_{n_days_to_predict}day_feature_cohort_importance.xlsx"), engine='openpyxl')
    print(df_features)

# filtered_list = [s for s in my_list if 'KTF1' not in s and 'KTF2' not in s]


# for model in models:
x_train, y1_train, y2_train, y3_train, model1, model2, model3 = training_pipline(
    X_train, y1_train, y2_train, y3_train, over_sampling_method)
y1_prob_all, y2_prob_all, y3_prob_all = evaluate_models(model,
                                                        model1, model2, model3, X_test, y1_test, y2_test, y3_test)
# cross_validation(model, x_train, X_test, model1, model2, model3,
#                     y1_train, y2_train, y3_train, n_cv_fold)

# plot_prediction_vs_true(y1, 1, n_days_to_predict)
# plot_prediction_vs_true(y2, 2, n_days_to_predict)
# plot_prediction_vs_true(y3, 3, n_days_to_predict)

# plot_prediction_vll, y2, y2_prob_all, y3, y3_prob_all)

# x_train is modified version of X_train by training pipline
# compute_shape_value(model, model1, x_train[0], X_train)
# generate_shap_feature_importance(model, model1)
