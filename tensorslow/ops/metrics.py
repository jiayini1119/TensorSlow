import numpy as np
from ..core import Node

class Metrics(Node):
    """
    parents[0]: prediction - probability, logit, or probability vector
    parents[1]: label - 1/-1 or one-hot
    """
    def __init__(self, *parents, **kwargs):
        kwargs['need_save'] = kwargs.get('need_save', False)
        Node.__init__(self, *parents, **kwargs)
        # Initialize the node
        self.init()    
    
    def reset(self):
        self.reset_value()
        self.init()    

    def init(self):
        pass    

    def get_jacobi(self):
        raise NotImplementedError()  
    
    def value_str(self):
        return "{}: {:.4f} ".format(self.__class__.__name__, self.value)

    def prob_to_label(prob, thresholds=0.5):
        """
        map logit/probability to label
        """
        if prob.shape[0] > 1:
            # multi-class
            labels = np.zeros((prob.shape[0], 1))
            labels[np.argmax(prob, axis=0)] = 1
        else:
            labels = np.where(prob < thresholds, -1, 1)
        return labels  

# confusion matrix
# TN   FP
# FN   TP

# Accuracy
class Accuracy(Metrics):
    # Accuracy = (TP + TN) / TOTAL

    def __init__(self, *parents, **kwargs):
        Metrics.__init__(self, *parents, **kwargs)

    def init(self):
        self.correct_num = 0
        self.total_num = 0

    def compute(self):
        pred = Metrics.prob_to_label(self.parents[0].value)
        gt = self.parents[1].value
        # TP + TN
        self.correct_num += np.sum(pred == gt)
        # TOTAL
        self.total_num += len(pred)
        self.value = 0
        if self.total_num != 0:
            self.value = float(self.correct_num) / self.total_num

# Precision
class Precision(Metrics):
    # Precision_p = TP / (TP + FP)
    # Precision_n = TN / (TN + FN)

    def __init__(self, *parents, **kwargs):
        Metrics.__init__(self, *parents, **kwargs)

    def init(self):
        self.true_pos_num = 0
        self.pred_pos_num = 0
    
    def compute(self):
        pred = Metrics.prob_to_label(self.parents[0].value)
        gt = self.parents[1].value
        # TP + FP (prediction is 1)
        self.pred_pos_num += np.sum(pred == 1)
        # TP (prediction is 1 && prediction is correct)
        self.true_pos_num += np.sum(pred == gt and pred == 1)
        self.value = 0
        if self.pred_pos_num != 0:
            self.value = float(self.true_pos_num) / self.pred_pos_num

# Recall
class Recall(Metrics):
    # Recall_p = TP / (TP + FN) => True Positive Rate
    # Recall_n = TN / (TN + FP) => True Negative Rate
    # False Positive Rate: FP / (TP + FN) == 1 - Recall_p
    def __init__(self, *parents, **kwargs):
        Metrics.__init__(self, *parents, **kwargs)

    def init(self):
        self.gt_pos_num = 0
        self.true_pos_num = 0

    def compute(self):
        pred = Metrics.prob_to_label(self.parents[0].value)
        gt = self.parents[1].value
        # TP + FN
        self.gt_pos_num += np.sum(gt == 1)
        # TP
        self.true_pos_num += np.sum(pred == gt and pred == 1)
        self.value = 0
        if self.gt_pos_num != 0:
            self.value = float(self.true_pos_num) / self.gt_pos_num 

# ROC - Receiver Operating Characteristic Curve
class ROC(Metrics):
    def __init__(self, *parents, **kwargs):
        Metrics.__init__(self, *parents, **kwargs)
    
    def init(self):
        self.count = 100
        self.gt_pos_num = 0
        self.gt_neg_num = 0
        self.true_pos_num = np.array([0] * self.count)
        self.false_pos_num = np.array([0] * self.count)
        self.tpr = np.array([0] * self.count)
        self.fpr = np.array([0] * self.count)

    def compute(self):

        prob = self.parents[0].value
        gt = self.parents[1].value
        self.gt_pos_num += np.sum(gt == 1)
        self.gt_neg_num += np.sum(gt == -1)

        # min = 0.01，max = 0.99，step = 0.01
        thresholds = list(np.arange(0.01, 1.00, 0.01))

        # run for each threshhold
        for index in range(0, len(thresholds)):
            pred = Metrics.prob_to_label(prob, thresholds[index])
            self.true_pos_num[index] += np.sum(pred == gt and pred == 1)
            self.false_pos_num[index] += np.sum(pred != gt and pred == 1)

        # calculate TPR and FPR
        if self.gt_pos_num != 0 and self.gt_neg_num != 0:
            self.tpr = self.true_pos_num / self.gt_pos_num
            self.fpr = self.false_pos_num / self.gt_neg_num
    
    def value_str(self):
        return ''


class ROC_AUC(Metrics):
    def __init__(self, *parents, **kargs):
        Metrics.__init__(self, *parents, **kargs)

    def init(self):
        self.gt_pos_preds = []
        self.gt_neg_preds = []

    def compute(self):
        prob = self.parents[0].value
        gt = self.parents[1].value
        # Assume only one element
        if gt[0, 0] == 1:
            self.gt_pos_preds.append(prob)
        else:
            self.gt_neg_preds.append(prob)

        self.total = len(self.gt_pos_preds) * len(self.gt_neg_preds)

    def value_str(self):
        count = 0
        for gt_pos_pred in self.gt_pos_preds:
            for gt_neg_pred in self.gt_neg_preds:
                if gt_pos_pred > gt_neg_pred:
                    count += 1

        self.value = float(count) / self.total
        return "{}: {:.4f} ".format(self.__class__.__name__, self.value)              