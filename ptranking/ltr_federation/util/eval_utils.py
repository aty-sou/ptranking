
import torch

from ptranking.data.data_utils import LABEL_TYPE
from ptranking.metric.adhoc.adhoc_metric import torch_ndcg_at_k

def compute_metric_score(batch_predict_rankings, batch_std_labels, k=10, presort=False, label_type=LABEL_TYPE.MultiLabel, device='cpu'):
    "Given the input ranker, compute the performance with pre-specified metrics "
    if presort:
        batch_ideal_rankings = batch_std_labels
    else:
        batch_ideal_rankings, _ = torch.sort(batch_std_labels, dim=1, descending=True)

    batch_ndcg_at_k = torch_ndcg_at_k(batch_predict_rankings=batch_predict_rankings,
                                      batch_ideal_rankings=batch_ideal_rankings,
                                      k=k, label_type=label_type, device=device)

    return batch_ndcg_at_k
