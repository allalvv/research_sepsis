"""
AUTHORS:
Team22_SepsisPrediction for Fall 2018 CSE6250

PURPOSE:
This script contains common utility functions used by all sepsis and septic shock prediction model scripts, including
data preprocessing functions and model performance metrics.
"""
import pandas as pd
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import glob
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import *
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from plots import roc_curve_plot, plot_confusion_matrix
import pickle
import distutils.dir_util

'''
# Includes the utility and commonly used methods by the different anlaysis class
'''

'''
# Read the csv file in the specified path
# input: directoryName, FolderName
# output: data
'''


def read_data_from_file(dirName, folderName):
    '''
    Read csv files and return dataframe.
    :return: pd.Dataframe
    '''
    path = '../../data/'+ dirName +'/' + folderName +'/' + '*.csv'
    for fname in glob.glob(path):
        data = pd.read_csv(fname)
    return data

'''
# LogisticRegression model with hyperparameters
# input: Xtrain, Xtest, Ytrain, number of iterations,type of analysis
# 
'''
def model_performance(Xtrain, Xtest, Ytrain, Ytest, k=5, randseed=545510477, analysis_type='sepsis1', balanced_class_weight=True):
    
    '''
    :param Xtrain, Xtest, Ytrain, Ytest: Train and test data
    :param k: k for K-fold cross-validation
    :param randseed: seed for randomizer
    :param analysis_type: String for type of analysis to be used as filename
    :return: None
    '''

    print("Model performance start")

    # Grid-search hyperparameter optimization
    # Create regularization hyperparameter space
    C = np.power(2, np.arange(0, 20, 2)) * 0.1
    logr = LogisticRegression(random_state=randseed,
                              max_iter=1000  # Use this if the solver doesn't converge; Increases processing time
                              )
    # Parameters to test for hyperparameter optimization
    if balanced_class_weight:
        class_weight_param = 'balanced'
    else:
        class_weight_param = None

    param_grid = {
        # 'pca__n_components': [2, 4, 6, 9], # comment out line if all features to be used
        'logr__C': C,
        'logr__penalty': ['l1', 'l2'],
        # 'logr__solver': ['newton-cg', 'lbfgs', 'sag'],  # for l2 penalty
        # 'logr__solver': ['liblinear','saga'], # for l1 penalty
        'logr__solver': ['newton-cg', 'lbfgs', 'liblinear', 'saga'],
        'logr__class_weight': [class_weight_param]
    }

    # Pipeline to optimize PCA and Logistic Regression parameters
    pca = PCA()
    pipe = Pipeline(steps=[('pca', pca), ('logr', logr)])
    clf = GridSearchCV(pipe, param_grid, cv=5, scoring='roc_auc', error_score=0.0, verbose=0)
    print("   GridSearchCV hyperparameter optimization start")
    clf.fit(Xtrain, Ytrain)
    print("   GridSearchCV hyperparameter optimization end")
    modelFolder = '../../output/models'
    distutils.dir_util.mkpath(modelFolder)
    filename = modelFolder + '/' + analysis_type + '_model.sav'

    # Save best model as file
    pickle.dump(clf, open(filename, 'wb'))

    # Using best performing parameters for LR classifier
    acc, auc_ = get_acc_auc_kfold(clf.best_estimator_, Xtrain, Ytrain, k=k)

    print("______________________________________________")
    print(("Classifier: Logistic Regression"))
    print("Best parameter (CV score=%0.3f):" % clf.best_score_)
    print(clf.best_params_)

    print(("Average Accuracy in KFold CV: %0.4f" % acc))
    print(("Average AUC in KFold CV:      %0.4f" % auc_))

    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(Ytest, clf.predict_proba(Xtest)[:, 1])
    roc_auc = auc(fpr, tpr)

    # Check model performance on test set
    Ytest_pred = clf.best_estimator_.predict(Xtest)
    acc = accuracy_score(Ytest, Ytest_pred)
    auc_ = roc_auc_score(Ytest, Ytest_pred)

    print(("Accuracy in Test set:         %0.4f" % acc))
    print(("AUC in Test set:              %0.4f" % auc_))
    print("")
    # print (Ytest_pred)

    class_names = ['Control', 'Cases']

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(Ytest, Ytest_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, analysis_type=analysis_type, normalize=False,
                          title='Confusion matrix, without normalization')
    print("______________________________________________")

    return fpr, tpr, roc_auc


'''
# Normalize the data
# input: Xtrain, Xtest
# output: Xtrain_norm, Xtest_norm
# 
'''


def normalize_X(Xtrain, Xtest):
    '''
    Normalize all X data to min-max of training set
    :param Xtrain: Training set of X
    :param Xtest: Test set of X
    :return:
    Xtrain_norm: Normalized Xtrain data using min-max of Xtrain
    Xtest_norm: Normalized Xtest data using min-max of Xtrain
    '''
    print("Normalization start")
    scaler = MinMaxScaler()
    Xtrain_norm = scaler.fit_transform(Xtrain)
    Xtest_norm = scaler.transform(Xtest)
    return Xtrain_norm, Xtest_norm


'''
# Print the best parameters 
# input: model
# 
'''


def print_clf_results(clf):
    '''
    Print classifier parameters and performance metrics
    :param clf: Classifier from Gridsearch cross-validation
    :return: None
    '''
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()


def general_clf_pred(clf, X_train, Y_train, X_test):
    '''
    Train classifier using training data and return predictions from test set.
    :param clf: Classifier
    :param X_train: Training set of X
    :param Y_train: Training Labels
    :param X_test: Test set of X
    :return: Predicted labels, Y_pred
    '''
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    return Y_pred


def get_acc_auc_kfold(clf, X, Y, k=5, randseed=545510477):
    '''
    Perform K-Fold cross-validation using input features (X) and labels (Y)
    :return: acc = mean accuracy, auc_ = mean AUC
    '''
    # First get the train indices and test indices for each iteration
    # Then train the classifier accordingly
    # Report the mean accuracy and mean auc of all the folds
    kf = KFold(n_splits=k, random_state=randseed)
    acc, auc_ = 0, 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        Y_pred = general_clf_pred(clf, X_train, Y_train, X_test)
        acc += accuracy_score(Y_test, Y_pred)
        auc_ += roc_auc_score(Y_test, Y_pred)
    acc = acc / k
    auc_ = auc_ / k

    return acc, auc_


'''
# Split the data into train and test
# input: Features, label
# output:Xtrain, Xtest, Ytrain, Ytest
# 
'''


def train_test_split(X, Y, test_size=0.3, randseed=545510477):
    '''
    Randomly sample X to split the data into train and test sets.
    :param X: features
    :param Y: labels
    :param test_size: The portion that should be used as test data
    :param randseed: seed for randomizer
    :return:
     Xtrain, Xtest, Ytrain, Ytest
    '''
    print("Train test split start")
    np.random.seed(randseed)
    msk = np.random.rand(len(Y)) < (1 - test_size)

    Xtrain = X[msk].astype(np.float64)
    Xtest = X[~msk].astype(np.float64)
    Ytrain = Y[msk].values
    Ytest = Y[~msk].values

    return Xtrain, Xtest, Ytrain, Ytest


'''
# Imputate the features as per paper that is grouped by age and gender for sepsis prediction
# input: Xtrain, Xtest, Ytrain, Ytest
# output:Xtrain, Xtest, Ytrain, Ytest
# 
'''


def sepsis_pred_imputation_age_gender(Xtrain, Xtest, Ytrain, Ytest):
    '''
    For sepsis features & labels, fill in missing values with means grouped by age and gender of the training data.
    
    :param Xtrain, Xtest, Ytrain, Ytest: Train and test data
    :return:
    Xtrain, Xtest, Ytrain, Ytest: Train and test data with missing values filled with means grouped by age group and gender
    '''
    print('Missing values train set: '+ str(Xtrain[Xtrain.isnull().any(axis=1)].shape[0]))
    print('Missing values test set: '+ str(Xtest[Xtest.isnull().any(axis=1)].shape[0]))
    print("Imputation using age group and gender")

    Xcols = ['admission_age','prev_hospital','CSRU_Service','MICU_Service'
            ,'SICU_Service','In_MICU','HeartRate','SysBP','RespRate'
            ,'TempC','weight','SOFA']

    # Get training data means by gender and age group
    Xtrain_gb = Xtrain.groupby(["gender", "age_bin"])
    Xtrain_means = Xtrain_gb.mean()

    # Get indexes that will be used for creating training means "filler" matrix
    Xtrain_idx4mean = Xtrain[["gender", "age_bin"]]
    Xtrain['means_idx'] = list(Xtrain_idx4mean.itertuples(index=False))
    Xtest_idx4mean = Xtest[["gender", "age_bin"]]
    Xtest['means_idx'] = list(Xtest_idx4mean.itertuples(index=False))

    # Create the means "filler" matrices for both training and test dateset
    Xtest_means_fill = Xtrain_means.loc[Xtest['means_idx'].values]
    Xtrain_means_fill = Xtrain_means.loc[Xtrain['means_idx'].values]

    # Label missing value locations using boolean mask; True=nan
    Xtrain_nas, Xtest_nas = Xtrain[Xcols].isna(), Xtest[Xcols].isna()
    Xtrain_index, Xtrain_vals = Xtrain.index, Xtrain[Xcols].values
    Xtest_index, Xtest_vals = Xtest.index, Xtest[Xcols].values

    # Fill in the missing values using training means
    Xtrain_vals[Xtrain_nas.values] = Xtrain_means_fill.values[Xtrain_nas.values]
    Xtest_vals[Xtest_nas.values] = Xtest_means_fill.values[Xtest_nas.values]

    # Recreate dataframes using imputed values
    Xtrain = pd.DataFrame(data=Xtrain_vals, index=Xtrain_index, columns=Xcols)
    Xtest = pd.DataFrame(data=Xtest_vals, index=Xtest_index, columns=Xcols)

    Xtrain, Xtest, Ytrain, Ytest = dropna_rows(Xtrain, Xtest, Ytrain, Ytest)
    return Xtrain, Xtest, Ytrain, Ytest


'''
# Drop rows that has null values
# input: Xtrain, Xtest, Ytrain, Ytest
# output:Xtrain, Xtest, Ytrain, Ytest
# 
'''


def dropna_rows(Xtrain, Xtest, Ytrain, Ytest):
    '''
    Drop all rows that have missing values.
    '''
    # Temporarily rejoin labels to feature dataframes
    Xtrain['Y'], Xtest['Y'] = Ytrain, Ytest
    Xtrain = Xtrain.dropna()
    Xtest = Xtest.dropna()

    # Re-separate the labels from the feature dataframes
    Ytrain = Xtrain['Y'].values
    Ytest = Xtest['Y'].values
    Xtrain = Xtrain.drop(labels=['Y'], axis=1)
    Xtest = Xtest.drop(labels=['Y'], axis=1)
    return Xtrain, Xtest, Ytrain, Ytest


'''
# Imputate all the features that is grouped by age and gender for sepsis prediction
# input: Xtrain, Xtest, Ytrain, Ytest
# output:Xtrain, Xtest, Ytrain, Ytest
# 
'''


# For testing different combination of features

def sepsis_pred_imputation_age_gender_experiment(Xtrain, Xtest, Ytrain, Ytest):
    '''
    For sepsis features & labels, fill in missing values with means grouped by age and gender of the training data.
    Experimental code.
    :param Xtrain, Xtest, Ytrain, Ytest: Train and test data
    :return:
    Xtrain, Xtest, Ytrain, Ytest: Train and test data with missing values filled with means grouped by age group and gender
    '''
    print('Missing values train set: '+ str(Xtrain[Xtrain.isnull().any(axis=1)].shape[0]))
    print('Missing values test set: '+ str(Xtest[Xtest.isnull().any(axis=1)].shape[0]))
    Xcols = ['admission_age','HeartRate','SysBP','DiasBP','MeanBP','RespRate',
            'TempC','SpO2','weight','SOFA','HeartRate_Min','HeartRate_Max',
            'HeartRate_Mean','SysBP_Min','SysBP_Max','SysBP_Mean','DiasBP_Min',
            'DiasBP_Max','DiasBP_Mean','MeanBP_Min','MeanBP_Max','MeanBP_Mean',
            'RespRate_Min','RespRate_Max','RespRate_Mean','TempC_Min','TempC_Max',
            'TempC_Mean','SpO2_Min','SpO2_Max','SpO2_Mean','Glucose_Min',
            'Glucose_Max','Glucose_Mean']

    # Xcols = ['admission_age', 'HeartRate', 'SysBP', 'DiasBP', 'MeanBP', 'RespRate',
    #          'TempC', 'SpO2', 'weight', 'SOFA', 'HeartRate_Min', 'HeartRate_Max',
    #          'HeartRate_Mean', 'SysBP_Min', 'SysBP_Max', 'SysBP_Mean', 'DiasBP_Min',
    #          'DiasBP_Max', 'DiasBP_Mean', 'MeanBP_Min', 'MeanBP_Max', 'MeanBP_Mean',
    #          'RespRate_Min', 'RespRate_Max', 'RespRate_Mean', 'TempC_Min', 'TempC_Max',
    #          'TempC_Mean', 'SpO2_Min', 'SpO2_Max', 'SpO2_Mean', 'Glucose_Min',
    #          'Glucose_Max', 'Glucose_Mean','ANIONGAP_min','ANIONGAP_max','ALBUMIN_min'
    #          ,'ALBUMIN_max','BANDS_min','BANDS_max','BICARBONATE_min','BICARBONATE_max'
    #             ,'BILIRUBIN_min','BILIRUBIN_max','CREATININE_min','CREATININE_max'
    #         ,'CHLORIDE_min','CHLORIDE_max','GLUCOSE_min','GLUCOSE_max'
    #         ,'HEMATOCRIT_min','HEMATOCRIT_max','HEMOGLOBIN_min','HEMOGLOBIN_max'
    #         ,'LACTATE_min','LACTATE_max','PLATELET_min','PLATELET_max','POTASSIUM_min'
    #         ,'POTASSIUM_max','PTT_min','PTT_max','INR_min','INR_max','PT_min','PT_max'
    #         ,'SODIUM_min','SODIUM_max','BUN_min','BUN_max','WBC_min','WBC_max']

    # Xcols = ['admission_age', 'HeartRate', 'SysBP', 'DiasBP', 'MeanBP', 'RespRate',
    #          'TempC', 'SpO2', 'weight', 'SOFA', 'HeartRate_Min', 'HeartRate_Max','SysBP_Min', 'SysBP_Max','DiasBP_Min',
    #          'DiasBP_Max','MeanBP_Min', 'MeanBP_Max','RespRate_Min', 'RespRate_Max','TempC_Min', 'TempC_Max','SpO2_Min', 'SpO2_Max']

    # Get training data means by gender and age group
    Xtrain_gb = Xtrain.groupby(["gender", "age_bin"])
    Xtrain_means = Xtrain_gb.mean()

    # Get indexes that will be used for creating training means "filler" matrix
    Xtrain_idx4mean = Xtrain[["gender", "age_bin"]]
    Xtrain['means_idx'] = list(Xtrain_idx4mean.itertuples(index=False))
    Xtest_idx4mean = Xtest[["gender", "age_bin"]]
    Xtest['means_idx'] = list(Xtest_idx4mean.itertuples(index=False))

    # Create the means "filler" matrices for both training and test dateset
    Xtest_means_fill = Xtrain_means.loc[Xtest['means_idx'].values]
    Xtrain_means_fill = Xtrain_means.loc[Xtrain['means_idx'].values]

    # Label missing value locations using boolean mask; True=nan
    Xtrain_nas, Xtest_nas = Xtrain[Xcols].isna(), Xtest[Xcols].isna()
    Xtrain_index, Xtrain_vals = Xtrain.index, Xtrain[Xcols].values
    Xtest_index, Xtest_vals = Xtest.index, Xtest[Xcols].values

    # Fill in the missing values using training means
    Xtrain_vals[Xtrain_nas.values] = Xtrain_means_fill.values[Xtrain_nas.values]
    Xtest_vals[Xtest_nas.values] = Xtest_means_fill.values[Xtest_nas.values]

    # Recreate dataframes using imputed values
    Xtrain = pd.DataFrame(data=Xtrain_vals, index=Xtrain_index, columns=Xcols)
    Xtest = pd.DataFrame(data=Xtest_vals, index=Xtest_index, columns=Xcols)

    Xtrain, Xtest, Ytrain, Ytest = dropna_rows(Xtrain, Xtest, Ytrain, Ytest)

    return Xtrain, Xtest, Ytrain, Ytest


'''
# Imputate the features as listed in the paper that is grouped by age and gender for septic shock prediction
# input: Xtrain, Xtest, Ytrain, Ytest
# output:Xtrain, Xtest, Ytrain, Ytest
# 
'''


def septic_shock_pred_imputation_age_gender(Xtrain, Xtest, Ytrain, Ytest):
    '''
    For septic shock features & labels, fill in missing values with means grouped by age and gender of the training data.

    :param Xtrain, Xtest, Ytrain, Ytest: Train and test data
    :return:
    Xtrain, Xtest, Ytrain, Ytest: Train and test data with missing values filled with means grouped by age group and gender
    '''
    print('Missing values train set: '+ str(Xtrain[Xtrain.isnull().any(axis=1)].shape[0]))
    print('Missing values test set: '+ str(Xtest[Xtest.isnull().any(axis=1)].shape[0]))
    print("Imputation using age group and gender")
    # for nan features calculate mean by age and gender
    ss_features_cols = ['HeartRate', 'HeartRate_Mean', 'SysBP', 'SysBP_Mean', 'PP', 'PP_Mean',
                        'RespRate', 'RespRate_Mean', 'TempC_Mean', 'SpO2', 'SpO2_Mean', 'WBC_min', 'WBC_max']

    # Get training data means by gender and age group
    Xtrain_gb = Xtrain.groupby(["gender", "age_bin"])
    Xtrain_means = Xtrain_gb.mean()

    # Get indexes that will be used for creating training means "filler" matrix
    Xtrain_idx4mean = Xtrain[["gender", "age_bin"]]
    Xtrain['means_idx'] = list(Xtrain_idx4mean.itertuples(index=False))
    Xtest_idx4mean = Xtest[["gender", "age_bin"]]
    Xtest['means_idx'] = list(Xtest_idx4mean.itertuples(index=False))

    # Create the means "filler" matrices for both training and test dateset
    Xtest_means_fill = Xtrain_means.loc[Xtest['means_idx'].values, ss_features_cols]
    Xtrain_means_fill = Xtrain_means.loc[Xtrain['means_idx'].values, ss_features_cols]

    # Label missing value locations using boolean mask; True=nan
    Xtrain_nas, Xtest_nas = Xtrain[ss_features_cols].isna(), Xtest[ss_features_cols].isna()
    Xtrain_index, Xtrain_vals = Xtrain.index, Xtrain[ss_features_cols].values
    Xtest_index, Xtest_vals = Xtest.index, Xtest[ss_features_cols].values

    # Fill in the missing values using training means
    Xtrain_vals[Xtrain_nas.values] = Xtrain_means_fill.values[Xtrain_nas.values]
    Xtest_vals[Xtest_nas.values] = Xtest_means_fill.values[Xtest_nas.values]

    # Recreate dataframes using imputed values
    Xtrain = pd.DataFrame(data=Xtrain_vals, index=Xtrain_index, columns=ss_features_cols)
    Xtest = pd.DataFrame(data=Xtest_vals, index=Xtest_index, columns=ss_features_cols)

    Xtrain, Xtest, Ytrain, Ytest = dropna_rows(Xtrain, Xtest, Ytrain, Ytest)

    return Xtrain, Xtest, Ytrain, Ytest


'''
# Imputate the features as listed in the paper using matrix factorization for septic shock prediction
# input: Xtrain, Xtest, Ytrain, Ytest
# output:Xtrain, Xtest, Ytrain, Ytest
# 
'''
def imputation_matrix_factorization(Xtrain, Xtest, Ytrain, Ytest):
    '''
    For sepsis features & labels, fill in missing values using matrix factorization.

    :param Xtrain, Xtest, Ytrain, Ytest: Train and test data
    :return:
    Xtrain, Xtest, Ytrain, Ytest: Train and test data with missing values filled
    '''
    print('Missing values train set: '+ str(Xtrain[Xtrain.isnull().any(axis=1)].shape[0]))
    print('Missing values test set: '+ str(Xtest[Xtest.isnull().any(axis=1)].shape[0]))
    print("imputation using matrix factorization")
    R_df = Xtrain.fillna(0)
    R = R_df.values
    user_ratings_mean = np.mean(R, axis=1)
    R_demeaned = R - user_ratings_mean.reshape(-1, 1)
    U, sigma, Vt = svds(R_demeaned, k=12)
    sigma = np.diag(sigma)
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    Xtrain = pd.DataFrame(all_user_predicted_ratings, columns=R_df.columns)

    R_df = Xtest.fillna(0)
    R = R_df.values
    user_ratings_mean = np.mean(R, axis=1)
    R_demeaned = R - user_ratings_mean.reshape(-1, 1)
    U, sigma, Vt = svds(R_demeaned, k=12)
    sigma = np.diag(sigma)
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    Xtest = pd.DataFrame(all_user_predicted_ratings, columns=R_df.columns)
    # drop gender column
    Xtrain = Xtrain.drop(['gender', 'age_bin'], axis=1)
    Xtest = Xtest.drop(['gender', 'age_bin'], axis=1)

    return Xtrain, Xtest, Ytrain, Ytest


'''
# split the data as case and control using the septic flag
# input: X, Y
# output:Xcases, Ycases, Xcontrols, Ycontrols
# 
'''
def case_control_split(X, Y):
    print("Case Control Split")
    septic_bin = Y == 1
    Xcases, Ycases = X[septic_bin], Y[septic_bin]
    Xcontrols, Ycontrols = X[~septic_bin], Y[~septic_bin]
    return Xcases, Ycases, Xcontrols, Ycontrols


'''
# LinearSVC model
#input: X_train, Y_train and X_test
#output: Y_pred
# 
'''
def svm_pred(X_train, Y_train, X_test):
    svm_model = LinearSVC(random_state=545510477, class_weight='balanced', C=1.0, tol=1e-5, fit_intercept=True,
                          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
                          multi_class='ovr', penalty='l2', )
    svm_model.fit(X_train, Y_train)
    Y_pred = svm_model.predict(X_test)
    return Y_pred


'''
# DecisionTree model
#input: X_train, Y_train and X_test
#output: Y_pred
# 
'''


def decisionTree_pred(X_train, Y_train, X_test):
    dt_model = DecisionTreeClassifier(max_depth=5, random_state=545510477, class_weight='balanced')
    dt_model.fit(X_train, Y_train)
    Y_pred = dt_model.predict(X_test)
    return Y_pred


'''
# Compute the metrics for the classifier
#input: Y_pred,Y_true
#output: accuracy, auc, precision, recall, f1-score
# 
'''


def classification_metrics(Y_pred, Y_true):
    accuracy = accuracy_score(Y_true, Y_pred)
    auc_score = roc_auc_score(Y_true.astype(int), Y_pred)
    precision = precision_score(Y_true, Y_pred)
    recall = recall_score(Y_true, Y_pred)
    f1 = f1_score(Y_true, Y_pred)
    return accuracy, auc_score, precision, recall, f1


'''
# Display the metrics to the console
#input: Name of classifier, predicted labels, actual labels
#output: accuracy, auc, precision, recall, f1-score
# 
'''


def display_metrics(classifierName, Y_pred, Y_true):
    print("______________________________________________")
    print(("Classifier: " + classifierName))
    acc, auc_, precision, recall, f1score = classification_metrics(Y_pred, Y_true)
    print(("Test Accuracy: " + str(acc)))
    print(("Test AUC: " + str(auc_)))
    # print(("Precision: "+str(precision)))
    # print(("Recall: "+str(recall)))
    # print(("F1-score: "+str(f1score)))
    print("______________________________________________")
    print("")
