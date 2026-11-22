"""The distillation objective F(w)."""
import torch


def kd_loss(student_logits, teacher_logits, targets, T: float, alpha: float):
    """Hinton-style knowledge-distillation loss (mean over the mini-batch):

        alpha * T^2 * KL( softmax(z_t/T) || softmax(z_s/T) )
      + (1 - alpha) * CE( y, softmax(z_s) )

    * The KL term makes the student copy the teacher's *softened* probabilities
      (temperature T > 1 exposes how the teacher ranks the wrong classes).
    * The T^2 factor keeps the gradient of the KL term on the same scale for
      every T (its gradient w.r.t. the logits is (1/T)*(p_s - p_t), times T^2 = T).
    * alpha = 0 gives plain label training (the "no distillation" baseline).

    KL and CE are written out explicitly with log-softmax for numerical
    stability instead of calling a library KD/KL function.
    """
    ce = -torch.log_softmax(student_logits, dim=1).gather(1, targets.unsqueeze(1)).mean()
    if alpha == 0.0 or teacher_logits is None:
        return ce
    log_ps = torch.log_softmax(student_logits / T, dim=1)
    log_pt = torch.log_softmax(teacher_logits / T, dim=1)
    kl = (log_pt.exp() * (log_pt - log_ps)).sum(dim=1).mean()
    return alpha * T * T * kl + (1.0 - alpha) * ce
