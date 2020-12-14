# Sepsis-3 Definition: How its Changes to Sepsis Clinical Criteria Impact the Performance of Preexisting Sepsis and Septic Shock Prediction Models
## Introduction
This repository contains the implementation for our research project for Georgia Tech's Online Masters in Computer Science (OMSCS) course [Big Data Analytics for Healthcare](http://www.sunlab.org/teaching/cse6250/fall2018/). The Spark SQL scripts run in Zeppelin notebook on a Apache Spark standalone cluster hosted on an Microsoft Azure Cloud Computing Platform virtual machine (Azure VM) with [MIMIC III data](http://mimic.physionet.org/).
This project aims to analyze ICU stays and observe the impact of Sepsis-3 criteria on preexisting sepsis and septic shock logistic regression prediction model. Please refer to the paper for a further explanation of the data source, methodology, model architecture, and results. Also, check the [video presentation](https://youtu.be/SShSo3dbKfg).

## Environment setup
1. Launch an Azure instance using step-by-step instructions on the Sun Lab website, [Option 2: Launch a clear linux](http://sunlab.org/teaching/cse6250/fall2018/env/env-azure-docker.html#launch-an-azure-instance).
   * Use a VM size of D4s_v3 or greater.
   * Use Ubuntu Server 16.04 LTS
   * During VM creation, under "INBOUND PORT RULES", select "Allow selected ports" for public inbound ports and select SSH.
   * Username=`mimicuser`, Password=`MIMIC III bd4h` 
2. [Connect to the instance](http://sunlab.org/teaching/cse6250/fall2018/env/env-azure-docker.html#connect-to-the-instance]) as indicated on the Sun Lab website.
   * After connecting, [Install Docker ](http://sunlab.org/teaching/cse6250/fall2018/env/env-local-docker-linux.html#install-docker-on-ubuntu-debian)
3. Using the ssh terminal session from the previou step, start the Sun Lab bigbox docker container
```bash
sudo docker run -it --privileged=true \
  --cap-add=SYS_ADMIN \
  -h bootcamp.local \
  --name bigbox -p 2222:22 -p 9530:9530 -p 8888:8888\
  -v /:/mnt/host \
  sunlab/bigbox:latest \
  /bin/bash
```
4. In ssh terminal session, execute `sudo docker attach bigbox`
   * This gives you command-line access to the docker container.
5. Execute `/scripts/start-services.sh`
6. Execute `/scripts/install-zeppelin.sh`
7. Execute `/scripts/start-zeppelin.sh`
8. Execute `/scripts/start-jupyter.sh`
1. Ctrl + P, Q to exit the docker container
1. Execute `exit` to exit the ssh session.

### To reconnect to Azure VM Zeppelin notebook
1. Using bash terminal, execute 
```bash
ssh -L 2222:localhost:2222 -L 8888:localhost:8888 -L 9530:localhost:9530 mimicuser@VM_IP_ADDRESS
```
where VM_IP_ADDRESS is your Azure VM's public IP address as shown in Azure portal. This step enables ssh port forwarding. When prompted for a password, use the password specified during VM creation.

2. Go to your internet browser (Google Chrome recommended) on your local machine. 
3. Open http://localhost:9530 which will load the Zeppelin notebook interface.

## Feature Construction using spark sql and Zeppelin notebook

Download the dataset from [MIMIC III](https://mimic.physionet.org/gettingstarted/dbsetup/) to a location on your environment.
The data preprocessing was accomplished using Zeppelin notebook. The first step is to pre-process the NOTEEVENTS.csv file. The text is multi-lined, and to be consumable by Spark SQL we need to convert the text for each note to be single lined. The python script will preprocess the NOTEEVENTS.csv to NOTEEVENTS_PROCESSED.csv. Upload this new file to a location where all the dataset from MIMIC III is located. Several notebooks are used to create intermediate tables and finally to extract useful features. Note that reference has been made to the SQL queries provided in this [Repo](https://github.com/MIT-LCP/mimic-code/tree/master/concepts) when constructing relevant features.

Run the scripts in the following order:
- Download the [MIMIC III](https://mimic.physionet.org/gettingstarted/dbsetup/) data and uncompress the files. Run the python script from the same location as the MIMIC III uncompressed files to convert the multi lined text in NOTEEVENTS.csv to be single line.
```bash
python processnotes.py
```
- All the zeppelin notebooks are under folder zeppelinNote
- Import the Zeppelin note sp_cohort.json from GitHub. This script will build and save vital and wbc observations, patient clinical history and ICU details, sepsis ICD-9 codes, the patients' first measurement of vitals during the first 6hr of stay in ICU as .csv files in the specified path. The script finally creates the cohort for sepsis1 and sepsis3 predictions.
- Import the Zeppelin note sofa_data_0_6.json from GitHub. This script will build and save various min and max measurements during the first 6hours in ICU required for SOFA(Sequential Organ Failure Assessment) to an output folder specified in the notebook.
- The Zeppelin notebook sofa_severityscore_0_6.json will calculate the SOFA severity scores using the measurements created by sofa_data_0_6.json.
- Import the Zeppelin note sofa_data_6_24.json from GitHub. This script will build and save various min and max measurements during the 6 - 24 hours in ICU required for SOFA(Sequential Organ Failure Assessment) to an output folder specified in the notebook.
- The Zeppelin notebook sofa_severityscore_6_24.json will calculate the SOFA severity scores using the measurements created by sofa_data_6_24.json.
- The Zeppelin notebook prescriptions.json defines patients suspected of infection for sepsis. Finally creates the dataset of patients with suspected infection and blood culture on the first day of their ICU admission.

## Predictive Modeling using python on local machine
This stage was deployed on the local machine using Python 3.6.5 environment. Use code/python/environment.yml which contains a list of libraries needed to set an environment for this project.
```bash
conda env create -f environment.yml
source activate cse6250project
```
- All the python code are under folder code/python. All the final features extracted by zeppelin notebook are stored under folder data. You can execute the python scripts directly without executing the zeppelin notebook.

| python script           |  Description   |
| :-----------------------|:---------------|
| sepsis1_analysis.py     | Read the .csv files created by Zeppelin notes, construct feature vectors, split into train and test and train the predictive models for sepsis1 prediction|
| sepsis3_analysis.py     | Read the .csv files created by Zeppelin notes, construct feature vectors, split into train and test and train the predictive models for sepsis-3 prediction|
| ss_analysis.py     |Read the .csv files created by Zeppelin notes, construct feature vectors, split into train and test and train the predictive models for septic shock prediction |
| run_all_analysis.py     |To run all analysis and generate charts |

### Example: Executing the scripts
```bash
python -W ignore run_all_analysis.py
```
Output:
```
----------- SEPSIS-1---------------
Get features and labels start
Train test split start
Missing values train set: 9015
Missing values test set: 2190
Imputation using age group and gender
Normalization start
Model performance start
   GridSearchCV hyperparameter optimization start
   GridSearchCV hyperparameter optimization end
______________________________________________
Classifier: Logistic Regression
Best parameter (CV score=0.834):
{'logr__C': 409.6, 'logr__class_weight': 'balanced', 'logr__penalty': 'l2', 'logr__solver': 'saga'}
Average Accuracy in KFold CV: 0.7487
Average AUC in KFold CV:      0.7585
Accuracy in Test set:         0.7427
AUC in Test set:              0.7461

Confusion matrix, without normalization
[[6260 2178]
 [ 243  730]]
______________________________________________
______________________________________________
Classifier: SVM
Test Accuracy: 0.7374349165869727
Test AUC: 0.7462889946059608
______________________________________________

______________________________________________
Classifier: Decision Tree
Test Accuracy: 0.7192646902560833
Test AUC: 0.7347924051305126
______________________________________________

Total Patients,          N = 49630
Training Sepsis-1 Cases, N = 4059
Training Controls,       N = 34339
Test Sepsis-1 Cases,     N = 973
Test Controls,           N = 8438

Processing Time: 00:01:26
----------- SEPSIS-3---------------
Get features and labels start
Train test split start
Missing values train set: 9015
Missing values test set: 2190
Imputation using age group and gender
Normalization start
Model performance start
   GridSearchCV hyperparameter optimization start
   GridSearchCV hyperparameter optimization end
______________________________________________
Classifier: Logistic Regression
Best parameter (CV score=0.849):
{'logr__C': 6553.6, 'logr__class_weight': 'balanced', 'logr__penalty': 'l1', 'logr__solver': 'saga'}
Average Accuracy in KFold CV: 0.7640
Average AUC in KFold CV:      0.7745
Accuracy in Test set:         0.7629
AUC in Test set:              0.7667

Confusion matrix, without normalization
[[6539 2041]
 [ 190  641]]
______________________________________________
______________________________________________
Classifier: SVM
Test Accuracy: 0.7587929019232813
Test AUC: 0.7655546018361903
______________________________________________

______________________________________________
Classifier: Decision Tree
Test Accuracy: 0.7481670385718839
Test AUC: 0.7570100477140188
______________________________________________

Total Patients,          N = 49630
Training Sepsis-3 Cases, N = 3498
Training Controls,       N = 34900
Test Sepsis-3 Cases,     N = 831
Test Controls,           N = 8580

Processing Time: 00:01:31
----------- SEPTIC SHOCK---------------
Train test split start
Missing values train set: 1095
Missing values test set: 236
Imputation using age group and gender
Normalization start
Train test split start
Missing values train set: 1095
Missing values test set: 236
Imputation using age group and gender
Normalization start
Sepsis-1 (septic shock)
Model performance start
   GridSearchCV hyperparameter optimization start
   GridSearchCV hyperparameter optimization end
______________________________________________
Classifier: Logistic Regression
Best parameter (CV score=0.739):
{'logr__C': 1.6, 'logr__class_weight': 'balanced', 'logr__penalty': 'l2', 'logr__solver': 'saga'}
Average Accuracy in KFold CV: 0.7111
Average AUC in KFold CV:      0.7141
Accuracy in Test set:         0.7335
AUC in Test set:              0.6725

Confusion matrix, without normalization
[[1020  330]
 [  85  122]]
______________________________________________
______________________________________________
Classifier: SVM
Test Accuracy: 0.7180475272960822
Test AUC: 0.6799355877616748
______________________________________________

______________________________________________
Classifier: Decision Tree
Test Accuracy: 0.6608863198458574
Test AUC: 0.6989079102715466
______________________________________________

Total Eligible Patients,     N = 8289
Sepsis-1 Septic Shock Cases, N = 207
Controls (Sepsis-1),         N = 1350

Sepsis-3 (septic shock)
Model performance start
   GridSearchCV hyperparameter optimization start
   GridSearchCV hyperparameter optimization end
______________________________________________
Classifier: Logistic Regression
Best parameter (CV score=0.810):
{'logr__C': 1.6, 'logr__class_weight': 'balanced', 'logr__penalty': 'l2', 'logr__solver': 'liblinear'}
Average Accuracy in KFold CV: 0.7659
Average AUC in KFold CV:      0.7443
Accuracy in Test set:         0.7913
AUC in Test set:              0.7732

Confusion matrix, without normalization
[[1153  299]
 [  26   79]]
______________________________________________
______________________________________________
Classifier: SVM
Test Accuracy: 0.7906229929351317
Test AUC: 0.781719795356159
______________________________________________

______________________________________________
Classifier: Decision Tree
Test Accuracy: 0.6608863198458574
Test AUC: 0.6989079102715466
______________________________________________

Total Eligible Patients,     N = 8289
Sepsis-3 Septic Shock Cases, N = 105
Controls (Sepsis-3),         N = 1452
```
The optimal model parameters and performance will be displayed. 
Optimized model(s) will be saved in the `output/model` folder.
