"""
AUTHORS:
Team22_SepsisPrediction for Fall 2018 CSE6250

PURPOSE:
This script performs cohort identification and feature extraction for sepsis prediction using Sepsis-3 criteria.
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
output: sofa_0_6hr, sofa_6_24hr, weight_firstday, sepsis1_features, sepsisByprescription
"""


def read_csv():
    '''
    Read the sofa scores, weight data, vitals metrics, sepsis patient data, and prescription information from csv files.
    Variables returned from this function are passed as input to the get_features_labels function.
    '''

    dirName = 'sofa_0_6'
    sofa_0_6hr = utils.read_data_from_file(dirName, 'sofa_0_6hr')
    weight_firstday = utils.read_data_from_file(dirName, 'weight_firstday')

    dirName = 'sofa_6_24'
    sofa_6_24hr = utils.read_data_from_file(dirName, 'sofa_6_24hr')

    dirName = 'sp_cohort'
    sepsis1_features = utils.read_data_from_file(dirName, 'sepsis1_features')
    sepsisByprescription = utils.read_data_from_file(dirName, 'sepsisByprescription')
    return sofa_0_6hr, sofa_6_24hr, weight_firstday, sepsis1_features, sepsisByprescription


"""
# Create features for the model:
###############################################################

# 'icustay_id'- ID defines a single ICU stay.
# 'gender' - genotypical sex of the patient. (M/F)
# 'admission_age' - age of the patient upon admission.
# 'first_hosp_stay' - define if the patient had previous hospital stay.(True/False)
# 'CSRU_Service'  - Cardiac surgery recovery unit flag if patient has service history.
# 'MICU_Service' - Medical intensive care unit flag if patient has service history.
# 'SICU_Service' - Surgical intensive care unit flag if patient has service history.
# 'sepsis1_icd' - Flag for sepsis ICD-9 diagnosis code (0 or 1)
#
# First observation for the first 6 hours of physiological state is collected
# ---------------------------------------------------------------------------
# 'HeartRate' - Heart Rate of the patient.
# 'SysBP' - Systolic Blood pressure of the patient.
# 'RespRate' - Respiratory rate of the patient.
# 'TempC' - Temperature in Celsius of the patient.
# 'In_MICU' - The last unit before the patient was discharged
# 'weight' - Weight of the patient upon admission
# 'SOFA' - Sequential Organ Failure Assessment score
#  Label - Antibiotics, SOFA and Sepsis1 flag used for label
############################################################
"""
    


def get_features_labels(sofa_0_6, sofa_6_24, weightfirstday, sepsis1_features, sepsisByprescription):
    '''
    Extracts features from ICU data and labels case and control instances using the same method as
    the Ghosh 2012 target study, but applies Sepsis-3 criteria to label sepsis patients.

    :param sofa_0_6: SOFA scores for first 6 hours of ICU stay
    :param sofa_6_24: SOFA scores from 6 hours to 24 hours after ICU admission
    :param weightfirstday: Weight of patient at admission of ICU stay
    :param sepsis1_features: Contains vitals metrics, first observation measurement during first 6 hours
    :param sepsisByprescription: Contains information about antibiottics administration and culture draws
    :return:
     features: feature data for all ICU stays cases
     labels: Series of labels for cohort; 1=Cases, 0=Controls
    '''
    print("Get features and labels start")

    # select all features from from the target paper
    features = sepsis1_features.loc[:,
               ['icustay_id', 'gender', 'admission_age', 'first_hosp_stay', 'CSRU_Service', 'MICU_Service',
                'SICU_Service', 'In_MICU', 'sepsis1_icd', 'HeartRate', 'SysBP', 'RespRate', 'TempC']]

    # Index all data by ICU stay ID
    sofa_0_6.set_index('icustay_id', inplace=True)
    sofa_6_24.set_index('icustay_id', inplace=True)
    weightfirstday.set_index('icustay_id', inplace=True)
    sepsisByprescription.set_index('icustay_id', inplace=True)

    # Add clinical history features
    features['gender'] = features['gender'].apply({'M': 0, 'F': 1}.get)
    features['prev_hospital'] = (features['first_hosp_stay'].values == 0).astype(int)
    features.drop('first_hosp_stay', axis=1, inplace=True)
    features.reindex()
    features.set_index('icustay_id', inplace=True)

    features = features.join(weightfirstday, lsuffix='_features', rsuffix='_weightfirstday')
    sofa_feature = sofa_0_6.loc[sofa_0_6.index, 'SOFA']
    features = features.join(sofa_feature, lsuffix='_features', rsuffix='_sofa_feature')


    # Remove row if there are not enough features
    features = features.dropna(thresh=11)

    # If age is greater then 89, replace it with 100.
    #    The MIMIC III database says "all ages > 89 in the database were replaced with 300"
    features.ix[features.admission_age > 89, 'admission_age'] = 100

    # Age groups based on https://www.cms.gov/Research-Statistics-Data-and-Systems/Statistics-Trends-and-Reports/NationalHealthExpendData/Age-and-Gender.html
    # 0 - 18, 19 - 44, 45 - 64, 65 - 84, and 85 and over
    age_bins = [0, 44, 64, 79, 100]
    features['age_bin'] = pd.cut(features['admission_age'], age_bins, labels=[0, 1, 2, 3], right=True)

    # Sepsis-3 case labelling
    sofa_6_24['sofa_change_gt_1'] = sofa_6_24['SOFA'].values > 1
    sepsisByprescription['has_antibiotics'] = (~(sepsisByprescription['antibiotic_name'].isna()))

    bool_features_sofa_gt_1 = sofa_6_24.loc[features.index, 'sofa_change_gt_1'].values
    bool_has_antibiotics = sepsisByprescription.loc[features.index, 'has_antibiotics'].values
    bool_sepsis1 = features.loc[features.index, 'sepsis1_icd'].values
    bool_sepsis3 = bool_sepsis1 & bool_features_sofa_gt_1 & bool_has_antibiotics

    features['sepsis3'] = bool_sepsis3.astype(np.int)
    labels = features['sepsis3'].copy()
    features = features.drop(labels=['sepsis3'], axis=1)
    features = features.drop(labels=['sepsis1_icd'], axis=1)

    return features, labels


# For testing different combination of features and additional data sets

def get_features_labels_Experiments(sp_cohort_icd, sofa_0_6, sofa_6_24, vitalsfirstday_0_6, sepsisByprescription, weightfirstday, sepsis1,labsfirstday):
    '''
    Extracts all available vitals and lab results to form features and labels.

    :param sofa_0_6: SOFA scores for first 6 hours of ICU stay
    :param sofa_6_24: SOFA scores from 6 hours to 24 hours after ICU admission
    :param vitalsfirstday_0_6: Vitals metrics for first size hours
    :param weightfirstday: Weight of patient at admission of ICU stay
    :param sepsis1_features: Contains vitals metrics, first observation measurement during first 6 hours
    :param sepsisByprescription: Contains information about antibiottics administration and culture draws
    :param labsfirstday: Lab results for first day
    :return:
     features: feature data for all ICU stays cases
     labels: Series of labels for cohort; 1=Cases, 0=Controls
    '''
    sepsis1.reindex()
    sepsis1.set_index('icustay_id', inplace=True)
    sepsis1.drop_duplicates(inplace=True)
    sofa_0_6.set_index('icustay_id', inplace=True)
    sofa_6_24.set_index('icustay_id', inplace=True)
    weightfirstday.set_index('icustay_id', inplace=True)
    sepsisByprescription.set_index('icustay_id', inplace=True)

    vitalsfirstday_0_6.set_index('icustay_id', inplace=True)
    vitalsfirstday_0_6 = vitalsfirstday_0_6.drop(['subject_id', 'hadm_id'], axis=1)

    labsfirstday.set_index('icustay_id', inplace=True)
    labsfirstday = labsfirstday.drop(['subject_id', 'hadm_id'], axis=1)


    # Features extraction

    features = sepsis1.drop(
        ['subject_id', 'hadm_id', 'intime', 'outtime', 'Glucose', 'admission_type', 'ethnicity', 'admittime',
         'dischtime', 'HeartRate_obs_count', 'SysBP_obs_count',
         'hospital_expire_flag', 'hospstay_seq', 'first_hosp_stay', 'los_icu', 'icustay_seq', 'first_icu_stay',
         'CSRU_Service', 'MICU_Service', 'SICU_Service', 'In_MICU', 'los_hospital',
         'DiasBP_obs_count', 'MeanBP_obs_count', 'RespRate_obs_count', 'TempC_obs_count', 'SpO2_obs_count',
         'WBC_observation_count', 'sepctic_shock_icd'], axis=1)

    features['gender'] = features['gender'].apply({'M': 0, 'F': 1}.get)
    features = features.join(weightfirstday, lsuffix='_features', rsuffix='_weightfirstday')
    sofa_feature = sofa_0_6.loc[sofa_0_6.index, 'SOFA']
    features = features.join(sofa_feature, lsuffix='_features', rsuffix='_sofa_feature')

    features = features.join(vitalsfirstday_0_6)
    features = features.join(labsfirstday)

    # Remove row if there are not enough features
    features = features.dropna(thresh=11)
    # If age is greater then 89, replace it with 100.
    #    The MIMIC III database says "all ages > 89 in the database were replaced with 300"
    features.ix[features.admission_age > 89, 'admission_age'] = 90

    # Age groups based on https://www.cms.gov/Research-Statistics-Data-and-Systems/Statistics-Trends-and-Reports/NationalHealthExpendData/Age-and-Gender.html
    # 0 - 18, 19 - 44, 45 - 64, 65 - 84, and 85 and over
    age_bins = [18, 44, 64, 84, 100]
    features['age_bin'] = pd.cut(features['admission_age'], age_bins, labels=[0, 1, 2, 3], right=True)

    # Sepsis-3 case labelling
    sofa_6_24['sofa_change_gt_1'] = sofa_6_24['SOFA'].values > 1
    sepsisByprescription['has_antibiotics'] = (~(sepsisByprescription['antibiotic_name'].isna()))

    bool_features_sofa_gt_1 = sofa_6_24.loc[features.index, 'sofa_change_gt_1'].values
    bool_has_antibiotics = sepsisByprescription.loc[features.index, 'has_antibiotics'].values
    bool_sepsis1 = features.loc[features.index, 'sepsis1_icd'].values
    bool_sepsis3 = bool_sepsis1 & bool_features_sofa_gt_1 & bool_has_antibiotics

    # features = features.groupby(["gender", "admission_age"]).transform(lambda x: x.fillna(x.mean()))
    # features = features.dropna()

    features['sepsis3'] = bool_sepsis3.astype(np.int)
    labels = features['sepsis3'].copy()
    features = features.drop(labels=['sepsis3'], axis=1)
    features = features.drop(labels=['sepsis1_icd'], axis=1)
    print(features.columns.values)
    return features, labels


"""
# Run sepsis septic shock analisys
###############################################################

input: 
output: fpr, tpr, roc_auc - ROC curves and ROC areas for each prediction
"""


def run_sepsis3():
    print("----------- SEPSIS-3---------------")
    # Load csv files
    sofa_0_6hr, sofa_6_24hr, weight_firstday, sepsis1_features, sepsisByprescription = read_csv()

    # Calculate sepsis features and labels for sepsis-3 criteria
    X, Y = get_features_labels(sofa_0_6hr, sofa_6_24hr, weight_firstday, sepsis1_features, sepsisByprescription)
    Xtrain, Xtest, Ytrain, Ytest = utils.train_test_split(X, Y, test_size=0.2)
    Xtrain, Xtest, Ytrain, Ytest = utils.sepsis_pred_imputation_age_gender(Xtrain, Xtest, Ytrain, Ytest)
    # Xtrain, Xtest, Ytrain, Ytest = utils.dropna_rows(Xtrain, Xtest, Ytrain, Ytest)
    Xtrain, Xtest = utils.normalize_X(Xtrain, Xtest)

    outputFolder = '../../output'
    distutils.dir_util.mkpath(outputFolder)

    # Save normalized train and test data
    np.savetxt(outputFolder+'/sepsis3_Xtrain.csv',Xtrain, delimiter=',')
    np.savetxt(outputFolder+'/sepsis3_Xtest.csv',Xtest, delimiter=',')
    np.savetxt(outputFolder+'/sepsis3_Ytrain.csv', Ytrain, delimiter=',')
    np.savetxt(outputFolder+'/sepsis3_Ytest.csv', Ytest, delimiter=',')

    # Print model performance
    start_time = time.time()
    fpr, tpr, roc_auc = utils.model_performance(Xtrain, Xtest, Ytrain, Ytest,
                                                k=5,
                                                randseed=RANDOM_STATE,
                                                analysis_type='sepsis3',
                                                # True: LogR class_weight='balanced' | False: LogR class_weight=None
                                                balanced_class_weight=True)
    utils.display_metrics("SVM", utils.svm_pred(Xtrain, Ytrain, Xtest), Ytest)
    utils.display_metrics("Decision Tree", utils.decisionTree_pred(Xtrain, Ytrain, Xtest), Ytest)
    elapsed_time = time.time() - start_time

    print("Total Patients,          N = " + str(sepsis1_features.shape[0]))
    print("Training Sepsis-3 Cases, N = " + str(np.size(Ytrain[Ytrain == 1])))
    print("Training Controls,       N = " + str(np.size(Ytrain[Ytrain == 0])))
    print("Test Sepsis-3 Cases,     N = " + str(np.size(Ytest[Ytest == 1])))
    print("Test Controls,           N = " + str(np.size(Ytest[Ytest == 0])))
    print("")
    print("Processing Time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    return fpr, tpr, roc_auc
