"""Pairwise ranking and description-set coverage objective for V5."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DescriptionSetCoverageLoss(nn.Module):
    """Pairwise ranking plus coverage over captions of the same image.


    A grouped training batch contains multiple human descriptions for every
    image identity.  The ordinary diagonal ranking term keeps the precise
    image-caption pair supervision.  The set term uses a smooth weakest-
    positive score, so every visual view must also cover the other valid
    descriptions in its group.  Samples carrying the same image id are never
    treated as negatives.
    """

    def __init__(
        self,
        margin=0.2,
        max_violation=False,
        coverage_weight=0.2,
        temperature=0.1,
    ):
        super().__init__()
        if coverage_weight < 0:
            raise ValueError("coverage_weight must be non-negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.margin = margin
        self.max_violation = max_violation
        self.coverage_weight = coverage_weight
        self.temperature = temperature
        self.last_pair_loss = None
        self.last_coverage_loss = None

    def max_violation_on(self):
        self.max_violation = True

    def max_violation_off(self):
        self.max_violation = False

    @staticmethod
    def _masked_extreme(values, mask, dim, temperature, largest):
        logits = values if largest else -values
        logits = (logits / temperature).masked_fill(~mask, -torch.inf)
        weights = torch.softmax(logits, dim=dim)
        safe_values = values.masked_fill(~mask, 0)
        return (weights * safe_values).sum(dim=dim)



    def _pairwise_loss(self, scores, positive_mask):
        diagonal = scores.diag().view(scores.size(0), 1)
        cost_i2t = (self.margin + scores - diagonal).clamp_min(0)
        cost_t2i = (self.margin + scores - diagonal.t()).clamp_min(0)
        cost_i2t = cost_i2t.masked_fill(positive_mask, 0)
        cost_t2i = cost_t2i.masked_fill(positive_mask, 0)
        if self.max_violation:
            cost_i2t = cost_i2t.max(dim=1).values
            cost_t2i = cost_t2i.max(dim=0).values
        return cost_i2t.sum() + cost_t2i.sum()

    def _coverage_direction(self, scores, positive_mask):
        negative_mask = ~positive_mask
        # Smooth-min over all valid descriptions: a low-scoring positive cannot
        # be hidden by an easier caption from the same image.
        weakest_positive = self._masked_extreme(
            scores,
            positive_mask,
            dim=1,
            temperature=self.temperature,
            largest=False,
        )
        strongest_negative = self._masked_extreme(
            scores,
            negative_mask,
            dim=1,
            temperature=self.temperature,
            largest=True,
        )
        return (
            self.margin + strongest_negative - weakest_positive
        ).clamp_min(0).sum()

    def forward(self, scores, img_ids, coverage_scale=1.0):
        if img_ids is None:
            raise ValueError("Description-set coverage requires image ids")
        if scores.ndim != 2 or scores.size(0) != scores.size(1):
            raise ValueError("Training similarity matrix must be square")
        if img_ids.numel() != scores.size(0):
            raise ValueError(
                "Image-id count must match the training similarity matrix"
            )

        img_ids = img_ids.reshape(-1)
        positive_mask = img_ids[:, None].eq(img_ids[None, :])
        pair_loss = self._pairwise_loss(scores, positive_mask)


        repeated = positive_mask.sum(dim=1) > 1
        if repeated.any():
            coverage_i2t = self._coverage_direction(scores, positive_mask)
            coverage_t2i = self._coverage_direction(
                scores.t(), positive_mask.t()
            )
            coverage_loss = coverage_i2t + coverage_t2i
        else:
            coverage_loss = scores.new_zeros(())
        self.last_pair_loss = pair_loss.detach()
        self.last_coverage_loss = coverage_loss.detach()
        return (
            pair_loss
            + float(coverage_scale) * self.coverage_weight * coverage_loss
        )
