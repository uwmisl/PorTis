import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_curve, auc
from itertools import cycle

from matplotlib_venn import venn3, venn3_circles, venn3_unweighted

def randargmax(b,**kw):
    np.random.seed(0)
    return np.argmax(np.random.random(b.shape) * (b==np.amax(b,**kw, keepdims=True)), **kw)

def create_KNN(k_neighbors):
    """
    Initialize Sklearn's k_neighbors classifier
    """
    return NearestNeighbors(n_neighbors=k_neighbors, metric='precomputed')

def best_neighbors_value(accuracy_mean,auc_score_mean):
    best_accuracy = np.argmax(accuracy_mean)
    best_auc_score = np.argmax(auc_score_mean)
    return best_accuracy, best_auc_score

def accuracy_bysample_and_byneighbor(pred_class, true_class):
    """
    Calculates accuracies of predictions depending on test sample and prediction values
    
    Inputs:
        pred_class: (num sample, nums neighbors tested, num signals)
        true_class: (num sample, num signals)
    Returns:
        accuracies: (num sample, nums neighbors)
    """
    # expand true_class to (num sample, nums neighbors tested, num signals)
    true_reformat = np.repeat(true_class, pred_class.shape[1], axis=0).reshape(pred_class.shape)
    # get the means comparing the signals (axis 2)
    accuracies = np.mean(pred_class == true_reformat, axis=2)
    return accuracies

def accuracy_mean_std_byneighbor(accuracies):
    """
    Aggregates accuracies across samples tested, returning mean and std deviation
    
    Input:  
        accuracies: (num sample, nums neighbors tested)
    
    Returns:
        acc_mean: (nums neighbors tested)
        acc_std: (nums neighbors tested)
    """
    acc_mean = np.mean(accuracies, axis=0)
    acc_std = np.std(accuracies, axis=0)
    return acc_mean, acc_std


def accuracy_bysignumb_fixneigh(pred_class, pred_conf, true_class, num_signals_vals, get_sigthld_conf=False):
    """
    Calculates accuracy by number of included signals discarding lower confidence ones,
    returning them averaged and std over samples. This one is for predictions with only one neighbor
    
    Inputs:
        pred_class: (num sample, num signals)
        pred_conf: (num sample, num signals)
        true_class: (num sample, num signals)
        num_signals_vals: (list or array with number of signals to include)
        get_sigthld_conf: Boolean, include to get a third dictionary with the confidence at the thld signal for each sample 

    Returns:
        acc_mean_results (num_signals_vals)
        sigthld_conf (num_signals_vals): Thld confidence of the least cut signal
    """
    # the shape is (number of values of included signals, number of neighbors)
    acc_mean_results = np.empty(len(num_signals_vals))
    acc_std_results = np.empty(len(num_signals_vals))

    # indices to sort by decreasing order of confidences, maintaining dimensions
    sort_idx = np.argsort(pred_conf)[...,::-1]

    # sort both the predicted and true labels in decreasing order of confidence
    pred_sorted = np.take_along_axis(pred_class, sort_idx, axis=-1)
    true_sorted = np.take_along_axis(true_class, sort_idx, axis=-1)
    # optionally sort the prediction confidences too to get the thld
    if get_sigthld_conf:
        conf_sorted = np.take_along_axis(pred_conf, sort_idx, axis=-1)
        thld_conf = []

    for i, num_sig in enumerate(num_signals_vals):
        # consider only num_sig signals with the higher confidence to calculate accuracy
        accmean = np.mean(pred_sorted[...,:num_sig] == true_sorted[...,:num_sig])
        acc_mean_results[i] = accmean
        # optionally get the confidences at the thld signal
        if get_sigthld_conf:
            thld_conf.append(conf_sorted[...,num_sig])

    if get_sigthld_conf:
        return acc_mean_results, np.stack(thld_conf)
    else:
        return acc_mean_results
    
def accuracy_bysignumb_randomthld(pred_class, pred_conf, true_class, num_signals_vals, get_sigthld_conf=False):
    """
    Calculates accuracy by number of included signals discarding random ones. Only for one neighbor values.
    
    Inputs:
        pred_class: (num sample, num signals)
        pred_conf: (num sample, num signals)
        true_class: (num sample, num signals)
        num_signals_vals: (list or array with number of signals to include)
        get_sigthld_conf: Boolean, include to get a third dictionary with the confidence at the thld signal for each sample value 

    Returns:
        acc_mean_results (num_signals_vals)
        sigthld_conf (num_signals_vals): Thld confidence of the least cut signal
    """
    # the shape is (number of values of included signals, number of neighbors)
    acc_mean_results = np.empty(len(num_signals_vals))
    acc_std_results = np.empty(len(num_signals_vals))

    # MOD the difference here is that the indices are generated randomly, not by confidences
    # indices to sort randomly
    np.random.seed(42)
    sort_idx = np.arange(len(pred_conf))
    np.random.shuffle(sort_idx) # acts in place
    # sort_idx = np.argsort(pred_conf)[...,::-1]

    # sort both the predicted and true labels in decreasing order of confidence
    pred_sorted = np.take_along_axis(pred_class, sort_idx, axis=-1)
    true_sorted = np.take_along_axis(true_class, sort_idx, axis=-1)
    # optionally sort the prediction confidences too to get the thld
    if get_sigthld_conf:
        conf_sorted = np.take_along_axis(pred_conf, sort_idx, axis=-1)
        thld_conf = []

    for i, num_sig in enumerate(num_signals_vals):
        # consider only num_sig signals with the higher confidence to calculate accuracy
        accmean = np.mean(pred_sorted[...,:num_sig] == true_sorted[...,:num_sig])
        acc_mean_results[i] = accmean
        # optionally get the confidences at the thld signal
        if get_sigthld_conf:
            thld_conf.append(conf_sorted[...,num_sig])

    if get_sigthld_conf:
        return acc_mean_results, np.stack(thld_conf)
    else:
        return acc_mean_results

def accuracy_bysignumb_byneighbors(pred_class, pred_conf, true_class, num_signals_vals, get_sigthld_conf=False):
    """
    Calculates accuracy by number of included signals discarding lower confidence ones,
    returning them averaged and std over samples
    
    Inputs:
        pred_class: (num sample, nums neighbors tested, num signals)
        pred_conf: (num sample, nums neighbors tested, num signals)
        true_class: (num sample, num signals)
        num_signals_vals: (list or array with number of signals to include)
        get_sigthld_conf: Boolean, include to get a third dictionary with the confidence at the thld signal for each sample 

    Returns:
        acc_mean_results (num_signals_vals, num neighbors tested)
        acc_std_results (num_signals_vals, num neighbors tested)
        sigthld_conf ()
    """
    # the shape is (number of values of included signals, number of neighbors)
    acc_mean_results = np.empty((len(num_signals_vals), pred_class.shape[1]))
    acc_std_results = np.empty((len(num_signals_vals), pred_class.shape[1]))
    
    # reformat true labels as pred_class, expanding it from 
    # (num sample, num signals) to (num sample, nums neighbors tested, num signals)
    true_class_reformat = np.repeat(true_class, pred_class.shape[1], axis=0).reshape(pred_class.shape)

    # indices to sort by decreasing order of confidences, maintaining dimensions
    sort_idx = np.argsort(pred_conf)[...,::-1]

    # sort both the predicted and true labels in decreasing order of confidence
    pred_sorted = np.take_along_axis(pred_class, sort_idx, axis=-1)
    true_sorted = np.take_along_axis(true_class_reformat, sort_idx, axis=-1)
    # optionally sort the prediction confidences too to get the thld
    if get_sigthld_conf:
        conf_sorted = np.take_along_axis(pred_conf, sort_idx, axis=-1)
        thld_conf = []

    for i, num_sig in enumerate(num_signals_vals):
        # consider only num_sig signals with the higher confidence to calculate accuracy
        accuracy = np.mean(pred_sorted[...,:num_sig] == true_sorted[...,:num_sig], axis=2)
        accmean, accstd = accuracy_mean_std_byneighbor(accuracy)
        acc_mean_results[i] = accmean
        acc_std_results[i] = accstd
        # optionally get the confidences at the thld signal
        if get_sigthld_conf:
            thld_conf.append(conf_sorted[...,num_sig])

    if get_sigthld_conf:
        return acc_mean_results, acc_std_results, np.stack(thld_conf)
    else:
        return acc_mean_results, acc_std_results

def plot_acc_bythld_byneigh(acc_results, thld_values, k_neighbors_options, title, xtick_freq=1, ytick_freq=1):
    """
    Plots either the mean or the std calculated in accuracy_bythld_byneighbors
    
    acc_results is either acc_mean_results or acc_std_results, size (num thlds, num neighbors tested)
    
    """
    plot_ = sns.heatmap(acc_results, yticklabels=[round(thld,2) for thld in thld_values], xticklabels=k_neighbors_options)
    new_ticks = [i.get_text() for i in plot_.get_xticklabels()]
    plt.xticks(range(0, len(new_ticks), xtick_freq), new_ticks[::xtick_freq])
    new_ticks = [i.get_text() for i in plot_.get_yticklabels()]
    plt.yticks(range(0, len(new_ticks), ytick_freq), new_ticks[::ytick_freq])
    plt.xlabel('k: Number of neighbors')
    plt.ylabel('Thld of informative signals')
    plt.title(title)
    plt.show()
    

   
    
def train_test_dtw_knn_single_sample(
    dtw_mat, labels, num_classes, num_splits,
    k_neighbors_options, knn_weights, report_add_info=False):
    """

    Here use num_splits, not num_samples to determine into how many partitions data should be sliced
    
    Splits are done into train and test, no validation
    
    If report_add_info is True, will return a dictionary with:
      - "pred_class": matrix with the assigned class of each signal (integer representing the class)
                      [num sample, num k vals of neighbors tested, predicted class of each signal]
      - "pred_conf": confidence in the assigned value (probability of that class estimated from % of neighbors)
                     [num sample, num k vals of neighbors tested, confidence of each prediction]
      - "true_class": matrix with the correct label of each prediction (note one less dimension because it is independent of neighbors)
                      [num sample, correct labels]
    """
    
    num_k_neighbors_options = len(k_neighbors_options)
    max_neighbors = max(k_neighbors_options)

    # Create empty arrays to store:
    # For each train-test split (diff splits for test set),
    # for different number of nearest neighbors considered, the predicted probability of
    # each class for each event in the test set
    probs_memory = np.empty((num_splits,  # first dimension for num splits
                            num_k_neighbors_options,    # second dimension for the number of k values for neighbors tested
                             int(dtw_mat.shape[0]/num_splits), # third dimension is num events in the test split
                             np.unique(labels).shape[0])) # last dimension are the probabilities of each class
    # to keep correct signal class for test samples of each sample used as evaluation. They are all the same in this case, but just in case they might be different
    # this one is also returned when report_add_info=True
    real_test_lbl = np.empty((num_splits, # first dimension is to store results for each test sample used
                            int(dtw_mat.shape[0]/num_splits))) # second to store the labels for each signal. ASSUMES same number of signals per each sample
    print(num_splits, int(dtw_mat.shape[0]/num_splits), real_test_lbl.shape)
    # initialize objects to return additional information
    if report_add_info:
        all_pred_class = np.empty((num_splits,  # first dimension for num splits
                         num_k_neighbors_options,    # second dimension for the number of k values for neighbors tested
                         int(dtw_mat.shape[0]/num_splits))) # third dimension is num events in the test split
        all_pred_conf = np.empty((num_splits,  # first dimension for num splits
                         num_k_neighbors_options,    # second dimension for the number of k values for neighbors tested
                         int(dtw_mat.shape[0]/num_splits))) # third dimension is num events in the test split
        
        # accuracy of each class along the classes, will be averaged
        accuracy_per_class = np.empty((num_splits,
                                       num_classes,
                                       num_k_neighbors_options))
        f1_sc = np.empty((num_splits, num_k_neighbors_options))
        precision_sc = np.empty((num_splits, num_k_neighbors_options))
        recall_sc = np.empty((num_splits, num_k_neighbors_options))
        
    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)
    
    for cv_i, (train_index, test_index) in enumerate(skf.split(dtw_mat, labels)):
        print(f"Now evaluating with split {cv_i+1}/{num_splits}")
        X_train = dtw_mat[np.ix_(train_index, train_index)]
        y_train = labels[train_index]
        # test matrix is test vs train distances
        X_test = dtw_mat[np.ix_(test_index, train_index)]
        y_test = labels[test_index]
    
        # print(f' - Train shape: {X_train.shape},\n - Test shape: {X_test.shape}')
        # print(f' - Train lab shape: {y_train.shape},\n - Test lab shape: {y_test.shape}')
        real_test_lbl[cv_i] = y_test

        # Fit classifier
        clf = create_KNN(max_neighbors)
        clf.fit(X_train)

        # find k nearest training signals for each test signal, using the distance matrix
        # neighbors are returned in order (from smaller distance to largest distance)
        # Here, we get the distances from and indices of the closest `max_neighbors` neighbors in X_train,
        # for each datapoint in X_test
        # the shape of both is (sigals in X_test, max_neighbors)
        neigh_distances, neigh_indices = clf.kneighbors(X_test)

        # get the true labels of the predicted neighbors. Preserves the order
        # the shape is [sigals in X_test, neighbors label]
        pred_neigh_labels = np.take(y_train, neigh_indices)

        # print("Finished model training")

        # iterate for all selected number of neighbors
        for it_neig, k_neighbors in enumerate(k_neighbors_options):
            pred_probs = []
            if knn_weights == 'uniform':
                # calculates the number of neighbors of each class for each sample to get probability of each signal belonging to class
                # subset for max number of neighbors in the second dimension to consider only k neighbors in each loop
                for i,(labl,w) in enumerate(zip(pred_neigh_labels[:,:k_neighbors],neigh_distances[:,:k_neighbors])):
                    pred_probs.append(np.bincount(labl.astype('int64'),minlength=num_classes)/labl.shape[0])
                pred_probs = np.array(pred_probs)
            elif knn_weights == 'distance':
                # calculates the number of neighbors of each class for each sample and weights it by distance to get probability of each signal belonging to class
                for i,(labl,dis) in enumerate(zip(pred_neigh_labels[:,:k_neighbors],neigh_distances[:,:k_neighbors])):
                    # if the sum of the distances is 0 (has found one or more repeated signals)
                    if np.sum(dis) == 0:
                        pred_probs.append(np.bincount(labl.astype('int64'), minlength=num_classes) / labl.shape[0])
                    else:
                        # weight is the inverse of the distance, since greater distance means less weight
                        w = 1.0 / dis
                        pred_probs.append(np.bincount(labl.astype('int64'),minlength=num_classes,weights=w)/np.sum(w))
                # store results
                pred_probs = np.array(pred_probs)

            # store the probability of each class for each event in the test set (repeated for diff number of k_neighbors)
            probs_memory[cv_i, it_neig] = pred_probs

    # Now, we have the predicted probabilities of each class for each event in the test set
    # and the true class for each event in the test set.
    # So now compute metrics for results
    accuracy = np.empty((num_splits, num_k_neighbors_options))
    auc_score = np.empty((num_splits, num_k_neighbors_options))
    for cv_i in range(num_splits):
        #MOD Reload y_test corresponding to that split, or it would take the last ones!
        y_test = real_test_lbl[cv_i]
        one_hot_y_test = np.zeros((len(y_test), num_classes))
        one_hot_y_test[np.arange(len(y_test)), y_test.astype(int)] = 1
        for it_neig in range(num_k_neighbors_options):
            # assigned class is the one with highest probability (more neighbors)
            # use custom argmax function to randomly break ties
            # pred_class = np.argmax(probs_memory[cv_i, it_neig], axis=1)  
            pred_class = randargmax(probs_memory[cv_i, it_neig], axis=1)
            # predicted confidence is the probability of the most likely class
            pred_conf = np.max(probs_memory[cv_i, it_neig], axis=1)
            accuracy[cv_i, it_neig] = np.mean(pred_class == y_test)
            auc_score[cv_i, it_neig] = roc_auc_score(one_hot_y_test, probs_memory[cv_i, it_neig, :, :], multi_class='ovr')  # ovr: one-versus-rest
            # if extra info requested, keep track of pred class and confidences
            if report_add_info:
                all_pred_class[cv_i, it_neig] = pred_class
                all_pred_conf[cv_i, it_neig] = pred_conf
                f1_sc[cv_i, it_neig] = f1_score(y_test, pred_class, average="macro")
                precision_sc[cv_i, it_neig] = precision_score(y_test, pred_class, average="macro")
                recall_sc[cv_i, it_neig] = recall_score(y_test, pred_class, average="macro")
                # calculate accuracy per class
                for cl in range(num_classes):
                    accuracy_per_class[cv_i, cl, it_neig] = np.mean([pred_class[y_test == cl] == y_test[y_test == cl]])
                                  
    # mean values of accuracy and auc_score across splits
    accuracy_mean = np.mean(accuracy, axis=0)
    auc_score_mean = np.mean(auc_score, axis=0)
    accuracy_sd = np.std(accuracy, axis=0)
    auc_score_sd = np.std(auc_score, axis=0)
                                  
    if report_add_info:
        # compile extra info in a dictionary
        add_info = {
            "pred_class": all_pred_class,
            "pred_conf": all_pred_conf,
            "true_class": real_test_lbl,
            "accuracy_per_class": accuracy_per_class,
            "f1_sc": f1_sc,
            "precision_sc": precision_sc,
            "recall_sc": recall_sc,
        }
    else:
         add_info = None        
    
    return accuracy, auc_score, accuracy_mean, auc_score_mean, accuracy_sd, auc_score_sd, add_info



def knn_classifier_ontest(X_train, y_train, X_test, y_test, k_neighbors, num_classes, knn_weights='uniform'):
    #  fit classifier
    clf = create_KNN(k_neighbors)
    clf.fit(X_train)

    # find k nearest training signals for each test signal, using the distance matrix
    # neighbors are returned in order (from smaller distance to largest distance)
    # Here, we get the distances from and indices of the closest `max_neighbors` neighbors in X_train,
    # for each datapoint in X_test
    # the shape of both is (sigals in X_test, max_neighbors)
    neigh_distances, neigh_indices = clf.kneighbors(X_test)

    # get the true labels of the predicted neighbors. Preserves the order
    # the shape is [sigals in X_test, neighbors label]
    pred_neigh_labels = np.take(y_train, neigh_indices)


    pred_probs = []

    if knn_weights == 'uniform':
        # calculates the number of neighbors of each class for each sample to get probability of each signal belonging to class
        # subset for max number of neighbors in the second dimension to consider only k neighbors in each loop
        for i,(labl,w) in enumerate(zip(pred_neigh_labels,neigh_distances)):
            pred_probs.append(np.bincount(labl.astype('int64'),minlength=num_classes)/labl.shape[0])
        pred_probs = np.array(pred_probs)
    elif knn_weights == 'distance':
        # calculates the number of neighbors of each class for each sample and weights it by distance to get probability of each signal belonging to class
        for i,(labl,dis) in enumerate(zip(pred_neigh_labels,neigh_distances)):
            # if the sum of the distances is 0 (has found one or more repeated signals)
            if np.sum(dis) == 0:
                pred_probs.append(np.bincount(labl.astype('int64'), minlength=num_classes) / labl.shape[0])
            else:
                # weight is the inverse of the distance, since greater distance means less weight
                w = 1.0 / dis
                pred_probs.append(np.bincount(labl.astype('int64'),minlength=num_classes,weights=w)/np.sum(w))
        pred_probs = np.array(pred_probs)
    else:
        raise ValueError(f'knn_weights must be "uniform" or "distance", but got {knn_weights}')

    # store the probability of each class for each event in the test set (repeated for diff number of k_neighbors)
    probs_memory = pred_probs

    # Now, we have the predicted probabilities of each class for each event in the test set
    # and the true class for each event in the test set.
    # So now compute metrics for results
    one_hot_y_test = np.zeros((len(y_test), num_classes))
    one_hot_y_test[np.arange(len(y_test)), y_test.astype(int)] = 1

    # assigned class is the one with highest probability (more neighbors)
    # use custom argmax function to randomly break ties
    # pred_class = np.argmax(probs_memory, axis=1)  
    pred_class = randargmax(probs_memory, axis=1)
    # predicted confidence is the probability of the most likely class
    pred_conf = np.max(probs_memory, axis=1)
    accuracy = np.mean(pred_class == y_test)
    auc_score = roc_auc_score(one_hot_y_test, probs_memory, multi_class='ovr')  # ovr: one-versus-rest

    precision, recall, fscore, support = score(y_test, pred_class)

    # keep track of pred class and confidences
    probs_info = {
            "pred_class": pred_class,
            "pred_conf": pred_conf,
            "true_class": y_test,
            "probs_memory": probs_memory, 
            "one_hot_y_test": one_hot_y_test,
    }
    # also return metrics
    metrics = {
            "accuracy": accuracy,
            "auc_score": auc_score,
            "precision": precision,
            "recall": recall,
            "fscore": fscore,
            "support": support,
    }
    
    return probs_info, metrics


def plot_roc_curve(probs_memory, one_hot_y_test, num_classes, title):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(one_hot_y_test[:, i], probs_memory[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(one_hot_y_test.ravel(), probs_memory.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    lw = 2

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= num_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for i, color in zip(range(num_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

def cluster_venn(cluster_counts, labels_key, equal_weight=False, lines=False, savepath=None):
    # subset_order = ['0','1','0,1','2','0,2','1,2','0,1,2']
    subset_order = ['[0]','[1]','[0 1]','[2]','[0 2]','[1 2]','[0 1 2]']
    subset = []
    for sub in subset_order:
        if sub in cluster_counts.keys():
            subset.append(cluster_counts[sub])
        else:
            # not found so not present
            subset.append(0)

    subset = tuple(subset)

    if equal_weight:
        venn_function = venn3_unweighted
    else:
        venn_function = venn3
        
    venn_function(subsets = subset, set_labels = (labels_key[0],labels_key[1],labels_key[2]), alpha = 0.5)
    
    if lines:
        if equal_weight:
            venn3_circles(np.ones(len(subset_order)))
        else:
            # to add line all around
            venn3_circles(subsets = subset, linestyle='-', linewidth=1, color='k')
    if savepath is not None:
        plt.savefig(savepath,dpi=300, bbox_inches='tight')
        
    plt.show()