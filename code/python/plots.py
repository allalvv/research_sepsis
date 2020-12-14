import matplotlib.pyplot as plt
import numpy as np
import itertools

import distutils.dir_util

"""
This file is used to create plots : Confusion matrix and ROC curves
"""

"""
  #########################
  Plot confusion matrix
  #########################
"""


def plot_confusion_matrix(cm, classes,
                          analysis_type='sepsis',
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    # Create folder if missing
    chartsFolder = create_folder()
    filename = chartsFolder + '/' + analysis_type + '_confusion_matrix.png'

    # Save the plot
    plt.savefig(filename)


"""
  #########################
  Plot all ROC curves
  
  input: fpr, tpr, roc_auc - arrays of ROC curves and ROC areas for each prediction
  #########################
"""


def roc_curve_plot(fpr, tpr, roc_auc):
    plt.figure()

    label = ['Sepsis 1', 'Sepsis 3', 'Septic Shock (Sepsis 1)', 'Septic Shock (Sepsis 3)']
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'darkviolet']

    for i in range(len(roc_auc)):
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                 label='ROC curve of {0}'
                       ''.format(label[i]))
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

    chartsFolder = create_folder()
    filename = chartsFolder + '/' + 'ROC_curve.png'
    # Save the plot
    plt.savefig(filename)

    # Create folder


def create_folder():
    chartsFolder = '../../charts'
    distutils.dir_util.mkpath(chartsFolder)
    return chartsFolder
