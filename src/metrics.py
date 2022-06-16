import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, auc,
    f1_score, precision_score, recall_score, roc_curve, roc_auc_score
)
from sklearn.metrics import confusion_matrix


def get_test_metrics(y_test, y_pred, y_pred_proba, save_path=None):
    """
    If y_pred_proba (probability scores) are provided,
    then we can compute ROC AUC Score.
    If multiclass, specify y_test in shape (n_datapoints, n_classes) and
    specify y_pred_proba in shape of (n_datapoints, n_classes) with the
    probability of each class for each datapoint, with the probabilities
    for each class summing to 1.
    
    Returns
    -------
    auroc, aupr, f1_score  
    """
    # Create one-hot encoding
    num_classes = len(np.unique(y_test))
    one_hot_y_test = np.zeros((len(y_test), num_classes))
    one_hot_y_test[np.arange(len(y_test)), y_test.astype(int)] = 1
    # Compute metrics
    auroc = roc_auc_score(one_hot_y_test, y_pred_proba, multi_class="ovr")
    aupr = average_precision_score(one_hot_y_test, y_pred_proba)
    accuracy = accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average=None)
    return auroc, aupr, accuracy, f1

# We want a single plot with a separate ROC curve for each class.
# Take the average across all samples.
def plot_roc_curves(fprs, tprs, roc_aucs, plot_title, save_path=None):
    """
    
    for each class
        tprs = []
        aucs = []
        for each sample
            get y_test array and y_pred_proba_list for using this sample as
        plot the mean
        
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html

    Parameters
    ----------
    fprs : dict{int, dict{int, list}}
        mapping from sample id (id of sample used for testing) to another dictionary
            mapping from class id to a list of fprs to use for ROC curve
    tprs : dict{int, dict{int, list}}
        mapping from sample id (id of sample used for testing) to another dictionary
            mapping from class id to a list of tprs to use for ROC curve
    roc_aucs : dict{int, dict{int, list}}
        mapping from sample id (id of sample used for testing) to another dictionary
            mapping from class id to roc auc score
    save_path : Optional{str}
        Optionally, a string path to save the plot to
    """
    
    num_samples = len(fprs)
    num_classes = len(fprs[0])
    
    for class_id in range(num_classes):
        class_tprs = []
        class_aucs = []
        class_mean_fpr = np.linspace(0, 1, 100)
        for sample_id in range(num_samples):
            interp_tpr = np.interp(class_mean_fpr, fprs[sample_id][class_id], tprs[sample_id][class_id])
            interp_tpr[0] = 0.0
            class_tprs.append(interp_tpr)
            class_aucs.append(roc_aucs[sample_id][class_id])
        # Compute average across all samples
        mean_tpr = np.mean(class_tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(class_mean_fpr, mean_tpr)
        std_auc = np.std(class_aucs)
        # Add a line to the plot for this class
        plt.plot(
            class_mean_fpr,
            mean_tpr,
            label=r"Mean ROC for class %d (AUC = %0.2f $\pm$ %0.2f)" % (class_id, mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )
        # TODO: Add stdv
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(plot_title)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    

def plot_confusion_matrix(conf_mat, labels_key, save_path=None):
    df_conf_mat = pd.DataFrame(conf_mat, range(len(conf_mat)), range(len(conf_mat)), dtype=int)
    df_conf_mat.columns = list(labels_key.values())
    df_conf_mat.index = list(labels_key.values())
    sn.set(font_scale=0.8)
    sn.heatmap(df_conf_mat, annot=True, fmt="d", annot_kws={"size": 9})
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()