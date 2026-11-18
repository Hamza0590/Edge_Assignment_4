"""
Knowledge-distillation framework.

Supports
  1. Soft-target KD with teacher predictions (Hinton et al., 2015)
  2. Temperature-scaled softmax
  3. Combined hard-label + soft-label objective
  4. Feature-based distillation (FitNets-style hint learning, Romero et al., 2015)

Total loss for one batch:

    L = (1 - alpha) * CE(student_logits, y)                          <- hard labels
      +      alpha  * T^2 * KL( p_teacher^T || p_student^T )         <- soft labels
      +      beta   * L_feat( r(f_student), f_teacher )              <- intermediate features

where p^T = softmax(logits / T). The T^2 factor keeps the gradient magnitude of
the soft term comparable to the hard term when T changes (Hinton et al.).

Feature distillation: the student's last feature map (C_s x H_s x W_s) is
mapped by a learnable regressor r (1x1 conv + BN) to the teacher's channel
count (2048) and resized to the teacher's spatial size; L_feat is the squared
L2 distance between the per-sample L2-normalised feature tensors (bounded in
[0, 4], so it is scale-free and can't overwhelm the other two terms).
The regressor is only used during training and is discarded afterwards,
so the deployed student is unchanged.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def temperature_softmax(logits, T=1.0):
    """softmax(z / T): T > 1 produces a softer distribution exposing 'dark knowledge'."""
    return F.softmax(logits / T, dim=1)


def soft_target_loss(student_logits, teacher_logits, T):
    """T^2 * KL(teacher^T || student^T), averaged over the batch."""
    log_p_s = F.log_softmax(student_logits / T, dim=1)
    p_t = temperature_softmax(teacher_logits, T)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)


class FeatureHint(nn.Module):
    """Regressor that projects student features into the teacher's feature space."""

    def __init__(self, student_channels, teacher_channels):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Conv2d(student_channels, teacher_channels, 1, bias=False),
            nn.BatchNorm2d(teacher_channels))

    def forward(self, f_s, f_t):
        f_s = self.regressor(f_s)
        if f_s.shape[-2:] != f_t.shape[-2:]:
            f_s = F.adaptive_avg_pool2d(f_s, f_t.shape[-2:])
        a = F.normalize(f_s.flatten(1).float(), dim=1)
        b = F.normalize(f_t.flatten(1).float(), dim=1)
        return (a - b).pow(2).sum(1).mean()


class Distiller(nn.Module):
    """Wraps a (trainable) student and a (frozen) teacher and computes the KD loss."""

    def __init__(self, student, teacher, alpha=0.9, temperature=4.0, beta=1.0):
        super().__init__()
        self.student, self.teacher = student, teacher
        self.alpha, self.T, self.beta = alpha, temperature, beta
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()
        self.hint = FeatureHint(student.feature_channels, teacher.feature_channels) if beta > 0 else None

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()          # the teacher always stays in eval mode (frozen BN stats)
        return self

    def forward(self, x, y):
        with torch.no_grad():
            t_logits, t_feat = self.teacher(x, return_features=True)
        s_logits, s_feat = self.student(x, return_features=True)
        s_logits, t_logits = s_logits.float(), t_logits.float()

        hard = F.cross_entropy(s_logits, y)
        soft = soft_target_loss(s_logits, t_logits, self.T)
        loss = (1 - self.alpha) * hard + self.alpha * soft
        feat = torch.zeros((), device=x.device)
        if self.hint is not None:
            feat = self.hint(s_feat, t_feat)
            loss = loss + self.beta * feat
        return loss, s_logits, {"hard": hard.detach(), "soft": soft.detach(), "feat": feat.detach()}
