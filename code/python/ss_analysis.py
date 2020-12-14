"""
AUTHORS:
Team22_SepsisPrediction for Fall 2018 CSE6250

PURPOSE:
This script performs cohort identification and feature extraction for septic shock prediction using Sepsis-1 and Sepsis-3
criteria.
A machine learning model (logistic regression) is then trained using hyperparameter optimization.
The best model parameters, based on AUC score, are printed to the console output for display along with performance metrics.
"""
import time
import pandas as pd
import numpy as np
import utils
import distutils.dir_util

RANDOM_STATE = 545510477

"""
#Read data from csv files
###############################################################

input: 
output: 
"""


def read_csv():
    '''
    Read the sofa scores, weight data, vitals metrics, lab results csv files, sepsis patient data,
    septic shock data, and prescription information from csv files.
    Variables returned from this function are passed as input to the feature extraction functions.
    '''
    # Read in data from observation region (First 6 hours of ICU stay)
    dirName = 'sofa_0_6'
    sofa_0_6hr = utils.read_data_from_file(dirName, 'sofa_0_6hr')
    weight_firstday = utils.read_data_from_file(dirName, 'weight_firstday')
    vitalsfirstday_0_6 = utils.read_data_from_file(dirName, 'vitals_firstday')
    labsfirstday_0_6 = utils.read_data_from_file(dirName, 'labs_firstday')

    # Read in data from sepsis and septic shock onset region (6 to 24 hrs after ICU admission)
    dirName = 'sofa_6_24'
    sofa_6_24hr = utils.read_data_from_file(dirName, 'sofa_6_24hr')

    # Read in data related to septic shock prediction cohort
    dirName = 'sp_cohort'
    septic_shock = utils.read_data_from_file(dirName, 'sp_cohort_icd_ss')
    sepsis1_features = utils.read_data_from_file(dirName, 'sepsis1_features')
    sepsisByprescription = utils.read_data_from_file(dirName, 'sepsisByprescription')

    return septic_shock, vitalsfirstday_0_6, labsfirstday_0_6, sofa_0_6hr, sofa_6_24hr, weight_firstday, sepsis1_features, sepsisByprescription

"""
# Create features for the model:
###############################################################

# 'icustay_id'- ID defines a single ICU stay.
# 'gender' - genotypical sex of the patient. (M/F)
# 'admission_age' - age of the patient upon admission.
#
# First observation for the first 6 hours of physiological state is collected
# ---------------------------------------------------------------------------
# 'HeartRate' - Heart Rate of the patient.
# 'SysBP' - Systolic Blood pressure of the patient.
# 'PP' - Systolic minus Diastolic blood pressure 
# 'RespRate' - Respiratory rate of the patient.
# 'TempC' - Temperature in Celsius of the patient.
# 'SpO2' - SpO2 of the patient
# 'weight' - Weight of the patient upon admission
#
# Metrics for full 6 hour observation window
# ------------------------------------------
# 'HeartRate_Mean' - Mean HeartRate
# 'SysBP_Mean' - Mean systolic blood pressure
# 'PP_Mean' - Mean PP
# 'RespRate_Mean' - Mean respiratory rate
# 'TempC_Mean' - Mean temperature in Celcius of the patient
# 'SpO2_Mean' - Mean SpO2 of the patient
# 'WBC_min' - Minimum White blood cell count of the patient
# 'WBC_max' - Maximum White blood cell count of the patient

############################################################
"""

def get_features_labels(septic_shock, vitalsfirstday_0_6, labsfirstday_0_6, sofa_0_6, sofa_6_24, weightfirstday,
                        sepsis1_features, sepsisByprescription):
    '''
    Generate features dataframe and class label array using input data.
    '''
    # Specify the features that will be used for the model (and include values that will be used for imputation of missing values)
    vitals_features_cols = ['HeartRate_Mean', 'SysBP_Mean', 'DiasBP_Mean', 'MeanBP_Mean', 'RespRate_Mean', 'TempC_Mean',
                            'SpO2_Mean']
    sepsis1_features_cols = ['HeartRate', 'SysBP', 'RespRate', 'TempC', 'SpO2']
    ss_features_cols = ['admission_age', 'age_bin', 'gender', 'weight', 'HeartRate', 'HeartRate_Mean',
                        'SysBP', 'SysBP_Mean', 'PP', 'PP_Mean', 'RespRate', 'RespRate_Mean',
                        'TempC_Mean', 'SpO2', 'SpO2_Mean', 'WBC_min', 'WBC_max']

    # Index all data by ICU stay ID
    sofa_0_6.set_index('icustay_id', inplace=True)
    sofa_6_24.set_index('icustay_id', inplace=True)
    septic_shock.set_index('icustay_id', inplace=True)
    vitalsfirstday_0_6.set_index('icustay_id', inplace=True)
    labsfirstday_0_6.set_index('icustay_id', inplace=True)
    sepsis1_features.set_index('icustay_id', inplace=True)
    weightfirstday.set_index('icustay_id', inplace=True)
    sepsisByprescription.set_index('icustay_id', inplace=True)

    # Only keep ICU stays that have enough observations
    septic_shock_to_keep_bool = (septic_shock['HeartRate_obs_count'].values >= 10) & \
                                (septic_shock['SysBP_obs_count'].values >= 10) & \
                                (septic_shock['DiasBP_obs_count'].values >= 10) & \
                                (septic_shock['MeanBP_obs_count'].values >= 10) & \
                                (septic_shock['RespRate_obs_count'].values >= 10) & \
                                (septic_shock['TempC_obs_count'].values >= 10) & \
                                (septic_shock['SpO2_obs_count'].values >= 10) & \
                                (septic_shock['WBC_observation_count'].values >= 2)
    septic_shock.drop(septic_shock[~septic_shock_to_keep_bool].index, inplace=True)

    # Build features dataframe
    features = vitalsfirstday_0_6[vitals_features_cols].loc[septic_shock.index]
    features[sepsis1_features_cols] = sepsis1_features.loc[:, sepsis1_features_cols]
    features['gender'] = sepsis1_features.loc[features.index, 'gender'].apply({'M': 0, 'F': 1}.get)
    features['admission_age'] = septic_shock.loc[features.index, 'admission_age'].values
    features[['WBC_min', 'WBC_max']] = labsfirstday_0_6.loc[features.index, ['WBC_min', 'WBC_max']]
    features['weight'] = weightfirstday.loc[features.index, 'weight'].values

    # Identify septic shock patients based on hypotensive regions and ICD-9 codes
    label_col = 'is_septic'
    features[label_col] = ((septic_shock.loc[features.index, label_col].values == 1) | (
            septic_shock.loc[features.index, 'sepctic_shock_icd'].values == 1)).astype(np.int)

    # Identify which of those septic shock patients meet the Sepsis-3 criteria
    sofa_6_24['sofa_change_gt_1'] = sofa_6_24['SOFA'].values > 1
    sepsisByprescription['has_antibiotics'] = (~(sepsisByprescription['antibiotic_name'].isna()))

    bool_features_sofa_gt_1 = sofa_6_24.loc[features.index, 'sofa_change_gt_1'].values
    bool_has_antibiotics = sepsisByprescription.loc[features.index, 'has_antibiotics'].values
    bool_sepsis1 = sepsis1_features.loc[features.index, 'sepsis1_icd'].values
    bool_sepsis3 = bool_sepsis1 & bool_features_sofa_gt_1 & bool_has_antibiotics

    isseptic3_label_col = 'is_septic3'
    features[isseptic3_label_col] = (bool_sepsis3 & features[label_col].values).astype(np.int)

    # Remove row if there are not enough features
    features = features.dropna(thresh=14)

    # If age is greater then 89, replace it with 100.
    #    The MIMIC III database says "all ages > 89 in the database were replaced with 300"
    features.ix[features.admission_age > 89, 'admission_age'] = 90

    # Age groups based on https://www.cms.gov/Research-Statistics-Data-and-Systems/Statistics-Trends-and-Reports/NationalHealthExpendData/Age-and-Gender.html
    # 0 - 18, 19 - 44, 45 - 64, 65 - 84, and 85 and over
    age_bins = [18, 44, 64, 84, 100]
    features['age_bin'] = pd.cut(features['admission_age'], age_bins, labels=[0, 1, 2, 3], right=True)

    # Add remaining features
    features['sofa'] = sofa_0_6.loc[features.index, 'SOFA'].values
    features['PP'] = sepsis1_features.loc[features.index, 'SysBP'] - sepsis1_features.loc[features.index, 'DiasBP']
    features['PP_Mean'] = features['SysBP_Mean'] - features['DiasBP_Mean']

    sepsis1_ss_features = features.copy()[ss_features_cols]
    sepsis1_ss_labels = features.copy().loc[:, label_col]
    sepsis3_ss_features = features.copy()[ss_features_cols]
    sepsis3_ss_labels = features.copy().loc[:, isseptic3_label_col]

    return sepsis1_ss_features, sepsis3_ss_features, sepsis1_ss_labels, sepsis3_ss_labels


"""
# Run sepsis septic shock analisys
###############################################################

input: 
output: fpr, tpr, roc_auc - ROC curves and ROC areas for each prediction
"""


def run_septic_shock():
    print("----------- SEPTIC SHOCK---------------")

    # Load CSV files
    septic_shock, vitalsfirstday_0_6, labsfirstday_0_6, sofa_0_6hr, sofa_6_24hr, weight_firstday, sepsis1_features, sepsisByprescription = read_csv()

    # Calculate septic shock features and labels for sepsis-1 and sepsis-3 criteria
    sepsis1_ss_features, sepsis3_ss_features, sepsis1_ss_labels, sepsis3_ss_labels = get_features_labels(septic_shock,
                                                                                                         vitalsfirstday_0_6,
                                                                                                         labsfirstday_0_6,
                                                                                                         sofa_0_6hr,
                                                                                                         sofa_6_24hr,
                                                                                                         weight_firstday,
                                                                                                         sepsis1_features,
                                                                                                         sepsisByprescription)

    # Train-test split of Sepsis-1 septic shock data followed by imputation and min-max normalization
    Xtrain, Xtest, Ytrain, Ytest = utils.train_test_split(sepsis1_ss_features, sepsis1_ss_labels, test_size=0.2)
    Xtrain, Xtest, Ytrain0, Ytest0 = utils.septic_shock_pred_imputation_age_gender(Xtrain, Xtest, Ytrain, Ytest)
    # Xtrain, Xtest, Ytrain0, Ytest0 = utils.dropna_rows(Xtrain, Xtest, Ytrain, Ytest)
    Xtrain0, Xtest0 = utils.normalize_X(Xtrain, Xtest)

    outputFolder = '../../output'
    distutils.dir_util.mkpath(outputFolder)

    # Save normalized train and test data
    np.savetxt(outputFolder + '/sepsis1_septic_shock_Xtrain.csv', Xtrain0, delimiter=',')
    np.savetxt(outputFolder + '/sepsis1_septic_shock_Xtest.csv', Xtest0, delimiter=',')
    np.savetxt(outputFolder + '/sepsis1_septic_shock_Ytrain.csv', Ytrain0, delimiter=',')
    np.savetxt(outputFolder + '/sepsis1_septic_shock_Ytest.csv', Ytest0, delimiter=',')

    # Train-test split of Sepsis-3 septic shock data followed by imputation and min-max normalization
    Xtrain, Xtest, Ytrain, Ytest = utils.train_test_split(sepsis3_ss_features, sepsis3_ss_labels, test_size=0.2)
    Xtrain, Xtest, Ytrain, Ytest = utils.septic_shock_pred_imputation_age_gender(Xtrain, Xtest, Ytrain, Ytest)
    Xtrain, Xtest = utils.normalize_X(Xtrain, Xtest)

    # Save normalized train and test data
    np.savetxt(outputFolder + '/sepsis3_septic_shock_Xtrain.csv', Xtrain, delimiter=',')
    np.savetxt(outputFolder + '/sepsis3_septic_shock_Xtest.csv', Xtest, delimiter=',')
    np.savetxt(outputFolder + '/sepsis3_septic_shock_Ytrain.csv', Ytrain, delimiter=',')
    np.savetxt(outputFolder + '/sepsis3_septic_shock_Ytest.csv', Ytest, delimiter=',')

    # Print model performance
    print('Sepsis-1 (septic shock)')
    fpr_s1, tpr_s1, roc_auc_s1 = utils.model_performance(Xtrain0, Xtest0, Ytrain0, Ytest0,
                                                         k=5,
                                                         randseed=RANDOM_STATE,
                                                         analysis_type='ss_sepsis1',
                                                         # True: LogR class_weight='balanced' | False: LogR class_weight=None
                                                         balanced_class_weight=True)
    utils.display_metrics("SVM", utils.svm_pred(Xtrain0, Ytrain0, Xtest0), Ytest0)
    utils.display_metrics("Decision Tree", utils.decisionTree_pred(Xtrain, Ytrain, Xtest), Ytest)

    print("Total Eligible Patients,     N = " + str(septic_shock.shape[0]))
    print("Sepsis-1 Septic Shock Cases, N = " + str(np.size(Ytest0[Ytest0 == 1])))
    print("Controls (Sepsis-1),         N = " + str(np.size(Ytest0[~(Ytest0 == 1)])))
    print("")
    print('Sepsis-3 (septic shock)')
    fpr_ss, tpr_ss, roc_auc_ss = utils.model_performance(Xtrain, Xtest, Ytrain, Ytest,
                                                         k=5,
                                                         randseed=RANDOM_STATE,
                                                         analysis_type='ss_sepsis3',
                                                         # True: LogR class_weight='balanced' | False: LogR class_weight=None
                                                         balanced_class_weight=True)
    utils.display_metrics("SVM", utils.svm_pred(Xtrain, Ytrain, Xtest), Ytest)
    utils.display_metrics("Decision Tree", utils.decisionTree_pred(Xtrain, Ytrain, Xtest), Ytest)
    print("Total Eligible Patients,     N = " + str(septic_shock.shape[0]))
    print("Sepsis-3 Septic Shock Cases, N = " + str(np.size(Ytest[Ytest == 1])))
    print("Controls (Sepsis-3),         N = " + str(np.size(Ytest[~(Ytest == 1)])))
    return fpr_s1, tpr_s1, roc_auc_s1, fpr_ss, tpr_ss, roc_auc_ss
