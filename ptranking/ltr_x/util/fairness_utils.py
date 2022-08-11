
import torch
import torch.nn.functional as F

EPS = 1e-20


def get_rank_span(batch_std_labels):
    # computing pairwise differences w.r.t. standard labels, i.e., S_{ij}
    batch_std_diffs = torch.unsqueeze(batch_std_labels, dim=2) - torch.unsqueeze(batch_std_labels, dim=1)
    # ensuring S_{ij} \in {-1, 0, 1}
    batch_Sij = torch.clamp(batch_std_diffs, min=-1.0, max=1.0)

    batch_indicator_l = batch_Sij < 0
    batch_indicator_le = batch_Sij <= 0

    batch_min_ranks = torch.sum(batch_indicator_l, dim=2) + 1.0
    batch_max_ranks = torch.sum(batch_indicator_le, dim=2)

    return batch_min_ranks, batch_max_ranks

def target_expected_exposure(batch_min_ranks, batch_max_ranks, gama):
    batch_mg = batch_max_ranks - batch_min_ranks + 1.0
    batch_target_epsilon = (torch.pow(gama, batch_min_ranks - 1.0) - torch.pow(gama, batch_max_ranks)) / \
                           (batch_mg * (1.0 - gama))
    return batch_target_epsilon

def exposure_gumbel_pertube(batch_preds, device='cpu'):
    unif = torch.rand(batch_preds.size(), device=device)  # [batch_size, ranking_size]
    gumbel = -torch.log(-torch.log(unif + EPS) + EPS)  # Sample from gumbel distribution
    batch_logits = batch_preds + gumbel
    batch_pertubed_probs = F.softmax(batch_logits, dim=1)
    return batch_pertubed_probs

def expected_exposure_objective(batch_sys_exposure, batch_target_exposure, balance_lambda):
    # Expected exposure disparity (EED) w.r.t. Equation-2
    EE_D = torch.sum(torch.sum(torch.square(batch_sys_exposure), dim=1))
    # Expected exposure relevance (EER): [batch_size, 1, ranking_size] x [batch_size, ranking_size, 1]
    EE_R = torch.bmm(torch.unsqueeze(batch_sys_exposure, dim=1), torch.unsqueeze(batch_target_exposure, dim=2))
    # Equation-6
    batch_loss = balance_lambda * EE_D - (1.0 - balance_lambda) * torch.sum(EE_R)

    return batch_loss