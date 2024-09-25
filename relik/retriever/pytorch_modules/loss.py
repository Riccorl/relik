from typing import Optional

import torch
from torch.nn.modules.loss import _WeightedLoss


class MultiLabelNCELoss(_WeightedLoss):
    __constants__ = ["reduction"]

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        size_average=None,
        reduction: Optional[str] = "mean",
    ) -> None:
        super(MultiLabelNCELoss, self).__init__(weight, size_average, None, reduction)

    def forward(
        self, input: torch.Tensor, target: torch.Tensor, ignore_index: int = -100
    ) -> torch.Tensor:

        valid_rows_mask = ~(target == -100)  # .all(dim=0)

        # Filter logits and labels based on valid rows
        input = input[valid_rows_mask]
        target = target[valid_rows_mask]

        gold_scores = input.masked_fill(~(target.bool()), 0)
        gold_scores_sum = gold_scores.sum(-1)  # B x C
        neg_logits = input.masked_fill(target.bool(), float("-inf"))  # B x C x L
        neg_log_sum_exp = torch.logsumexp(neg_logits, -1, keepdim=True)  # B x C x 1
        norm_term = (
            torch.logaddexp(input, neg_log_sum_exp)
            .masked_fill(~(target.bool()), 0)
            .sum(-1)
        )
        gold_log_probs = gold_scores_sum - norm_term
        loss = -gold_log_probs.sum()
        if self.reduction == "mean":
            loss /= input.size(0)
        return loss

    # def forward(
    #     self, input: torch.Tensor, target: torch.Tensor, ignore_index: int = -100
    # ) -> torch.Tensor:
    #     mask = target != ignore_index  # Positions that are not padding
    #     positive_mask = (target == 1) & mask  # Positive examples (target == 1)
    #     negative_mask = (target == 0) & mask  # Negative examples (target == 0)

    #     # Compute gold scores by selecting positive positions
    #     gold_scores = input.masked_fill(~positive_mask, 0)
    #     gold_scores_sum = gold_scores.sum(-1)  # Sum over the last dimension

    #     # Compute negative logits by selecting negative positions
    #     neg_logits = input.masked_fill(~negative_mask, float("-inf"))
    #     neg_log_sum_exp = torch.logsumexp(neg_logits, -1)

    #     # Compute the normalization term for positive positions
    #     norm_term = gold_scores_sum + neg_log_sum_exp

    #     # Compute the final loss
    #     gold_log_probs = gold_scores_sum - norm_term
    #     loss = -gold_log_probs.sum()
    #     if self.reduction == "mean":
    #         loss /= input.size(0)
    #     return loss
    # def forward(self, logits, labels):
    #     """
    #     logits: Tensor of shape (batch_size, num_passages) - the result of query * passages similarity (precomputed)
    #     labels: Tensor of shape (batch_size, num_passages) - 1 for positive passages, 0 for negative passages, -100 to ignore entire rows
    #     """

    #     # Create a mask for rows that should contribute to the loss (rows where labels are not all -100)
    #     valid_rows_mask = ~(labels == -100) #.all(dim=0)

    #     # Filter logits and labels based on valid rows
    #     valid_logits = logits[valid_rows_mask]
    #     valid_labels = labels[valid_rows_mask]

    #     # If there are no valid rows, return zero loss
    #     if valid_logits.size(0) == 0:
    #         return torch.tensor(0.0, requires_grad=True)

    #     # Ensure logits are in a range where softmax will work well
    #     logits_exp = torch.exp(valid_logits)

    #     # Positive passages are where labels are 1, negatives where labels are 0
    #     pos_mask = valid_labels == 1
    #     neg_mask = valid_labels == 0

    #     # Sum over positive and negative logits
    #     pos_logits_exp_sum = logits_exp * pos_mask  # Keep positive logits
    #     neg_logits_exp_sum = logits_exp * neg_mask  # Keep negative logits

    #     # Denominator: Sum of exponentiated positive and negative logits for each query
    #     denominator = pos_logits_exp_sum.sum(dim=0, keepdim=True) + neg_logits_exp_sum.sum(dim=0, keepdim=True)

    #     # Compute the loss: only consider positive passages (those where label is 1)
    #     # log( exp(sim_q_p_pos) / (sum of exp(sim_q_p_pos and sim_q_p_neg)) )
    #     pos_log_probs = torch.log(pos_logits_exp_sum.sum(dim=0) / denominator.squeeze(1))

    #     # Since we are calculating a loss, we want the negative of log probabilities for positive examples
    #     loss = -pos_log_probs.mean()

    #     return loss
