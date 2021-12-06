class Metric():
    def __init__(self, label, probabilities):
        prob_idx = [(probabilities[i], i) for i in range(len(probabilities))]
        prob_idx.sort(reverse=True)
        self.prob = [ele[0] for ele in prob_idx]
        idx = [ele[1] for ele in prob_idx]
        self.label = [label[i] for i in idx]
        self.size = len(label)

    def roc(self, threshold):
        idx = 0
        recalls = []
        precisions = []
        for thr in threshold:
            idx, tp, tn, fp, fn = self._compute_tp_tn_fp_fn(thr, idx)
            recall = self._compute_recall(tp, fn)
            precision = self._compute_precision(tp, fp)
            recalls.append(recall)
            precisions.append(precision)
        return recalls, precisions

    def pr(self, threshold):
        idx = 0
        tprs = []
        fprs = []
        for thr in threshold:
            idx, tp, tn, fp, fn = self._compute_tp_tn_fp_fn(thr, idx)
            tpr = self._compute_tpr(tp, fn)
            fpr = self._compute_fpr(fp, tn)
            tprs.append(tpr)
            fprs.append(fpr)
        return tprs, fprs


    def _compute_tp_tn_fp_fn(self, thr, last_idx):
        idx = -1
        for i in range(last_idx, self.size):
            if self.prob[i] < thr:
                idx = i - 1
                break
        tp, tn, fp, fn = 0, 0, 0, 0
        for j in range(self.size):
            if j <= idx:
                if self.label[j] == 1:
                    tp += 1
                else:
                    fp += 1
            else:
                if self.label[j] == 1:
                    fn += 1
                else:
                    tn += 1
        return idx, tp, tn, fp, fn

    def _compute_recall(self, tp, fn):
        return tp / (tp + fn)

    def _compute_precision(self, tp, fp):
        return tp / (tp + fp)

    def _compute_tpr(self, tp, fn):
        return tp / (tp + fn)

    def _compute_fpr(self, fp, tn):
        return fp / (fp + tn)
