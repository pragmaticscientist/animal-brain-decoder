from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score

def compute_metrics(true_labels, pred_labels):
    acc = accuracy_score(true_labels, pred_labels)
    prec = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
    rec = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
    f1_micro = f1_score(true_labels, pred_labels, average='micro', zero_division=0)
    r2 = r2_score(true_labels, pred_labels)

    return acc, prec, rec, f1_micro, r2