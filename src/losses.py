# Smooth student loss F(w), shared by all compression methods.
import torch


def kd_loss(student_logits, teacher_logits, targets, T: float, alpha: float):
    # Return mean batch loss: alpha*T^2*KL(teacher_T || student_T)
    # + (1-alpha)*CE(targets, student). alpha=0 gives label-only training.
    # log_softmax avoids unstable log(softmax(logits)) calculations.
    ce = -torch.log_softmax(student_logits, dim=1).gather(1, targets.unsqueeze(1)).mean()
    if alpha == 0.0 or teacher_logits is None:
        return ce
    log_ps = torch.log_softmax(student_logits / T, dim=1)
    log_pt = torch.log_softmax(teacher_logits / T, dim=1)
    kl = (log_pt.exp() * (log_pt - log_ps)).sum(dim=1).mean()
    return alpha * T * T * kl + (1.0 - alpha) * ce
