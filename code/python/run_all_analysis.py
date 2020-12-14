from plots import roc_curve_plot
from sepsis1_analysis import run_sepsis1
from sepsis3_analysis import run_sepsis3
from ss_analysis import run_septic_shock

"""
Main:  This file is used to run all 3 analyzis 
"""


def main():
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Run all models
    fpr[0], tpr[0], roc_auc[0] = run_sepsis1()
    fpr[1], tpr[1], roc_auc[1] = run_sepsis3()
    fpr[2], tpr[2], roc_auc[2], fpr[3], tpr[3], roc_auc[3] = run_septic_shock()

    # Creare ROC curve for all the predictions
    roc_curve_plot(fpr, tpr, roc_auc)


if __name__ == "__main__":
    main()
