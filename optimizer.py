import torch
import math
from torch import Tensor
from typing import List
from torch.nn import functional as F


def adam(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[int],
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
):
    """Implementation of the Adam Optimization"""
    with torch.no_grad():
        for i, param in enumerate(params):

            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step = state_steps[i]

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            if weight_decay != 0:
                grad = grad.add(param, alpha=weight_decay)

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(
                    eps
                )
            else:
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

            step_size = lr / bias_correction1

            param.addcdiv_(exp_avg, denom, value=-step_size)


def ini_alg(
    params: List[Tensor], grads: List[Tensor], L: float, lam: float, epsilon: float
):
    """Implementation of the Algorithm 2 in the paper"""
    with torch.no_grad():
        lr = 1 / L
        alpha = lam * epsilon / L

        for i, param in enumerate(params):
            grad = grads[i]
            zero = torch.zeros_like(grad)
            tmp = lr * grad.abs() - alpha
            param.add_(-1 * torch.max(tmp, zero) * grad.sign())


class ISTA(torch.optim.Optimizer):
    """Implementation of the proposed approach in which we first perform the Algorithm 2 to update the weights or params in our case,
    and then, perform the Adam algorithm on the updated point."""

    def __init__(self, parameters, defaults):
        super(ISTA, self).__init__(parameters, defaults)

    def __setstate__(self, state):
        super(ISTA, self).__setstate__(state)

    def step_(self, model, x, y, closure=None, multiplier=1):
        """the function performs a step of the optimization"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:  # iterate through the parameter groups
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_sums = []
            max_exp_avg_sqs = []
            state_steps = []
            epsilon = group["epsilon"]
            L = group["L"] / multiplier
            lam = group["lam"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                params_with_grad.append(p)
                grads.append(p.grad)
                state = self.state[
                    p
                ]  # by default we will have a stat:dict that records the current status of the associated parameters.

                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])

                state["step"] += 1
                state_steps.append(state["step"])

            ini_alg(params_with_grad, grads, L, lam, epsilon)

            self.zero_grad()
            y_hat = model(x)
            loss = F.cross_entropy(y_hat, y)
            loss.backward()

            adam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                False,
                0.9,
                0.999,
                1 / L,
                0.0,
                1e-8,
            )
        return loss


class QAT(torch.optim.Optimizer):
    """Implementation of QAT"""

    def __init__(self, parameters, defaults):
        super(QAT, self).__init__(parameters, defaults)

    def __setstate__(self, state):
        super(QAT, self).__setstate__(state)

    def step_(self):
        """the function performs a step of the optimization"""
        for group in self.param_groups:  # iterate through the parameter groups
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_sums = []
            max_exp_avg_sqs = []
            state_steps = []
            epsilon = group["epsilon"]
            L = group["L"]
            lam = group["lam"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                params_with_grad.append(p)
                grads.append(p.grad)
                state = self.state[
                    p
                ]  # by default we will have a stat:dict that records the current status of the associated parameters.

                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])

                state["step"] += 1
                state_steps.append(state["step"])

            adam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                False,
                0.9,
                0.999,
                1 / L,
                0.0,
                1e-8,
            )
        return
