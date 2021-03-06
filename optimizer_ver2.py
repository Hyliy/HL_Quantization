import torch
import math
from torch import Tensor
from typing import List, Optional
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


def sgd(params: List[Tensor],
        d_p_list: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        maximize: bool):
    r"""Functional API that performs SGD algorithm computation.
    See :class:`~torch.optim.SGD` for details.
    """
    with torch.no_grad():
        for i, param in enumerate(params):

            d_p = d_p_list[i]
            if weight_decay != 0:
                d_p = d_p.add(param, alpha=weight_decay)

            if momentum != 0:
                buf = momentum_buffer_list[i]

                if buf is None:
                    buf = torch.clone(d_p).detach()
                    momentum_buffer_list[i] = buf
                else:
                    buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                if nesterov:
                    d_p = d_p.add(buf, alpha=momentum)
                else:
                    d_p = buf

            alpha = lr if maximize else -lr
            param.add_(d_p, alpha=alpha)


def ini_alg(
        params: List[Tensor], grads: List[Tensor], L: float, lam: float, epsilon: float, multipler: float):
    """Implementation of the step 1"""
    with torch.no_grad():
        lr = multipler * 1 / L

        for i, param in enumerate(params):
            grad = grads[i]
            zero = torch.zeros_like(grad)
            temp_0 = (param / epsilon).round()
            temp_1 = param - temp_0 * epsilon
            temp_2 = epsilon / 2 - temp_1.abs()
            alpha = 2 / epsilon * lam * torch.max(zero, temp_2) / L

            if grad[grad.abs() > 0].size()[0] == 0:
                grad.add_(torch.randint(-1, 1, size=grad.size(), device='cuda') * 0.1 * epsilon)

            temp_3 = temp_1 - lr * grad
            tmp = temp_3.abs() - alpha
            param.add_(-temp_1 + torch.max(tmp, zero) * temp_3.sign())


class ISTA_LIU(torch.optim.Optimizer):
    """Implementation of ISTA"""

    def __init__(self, parameters, defaults):
        super(ISTA_LIU, self).__init__(parameters, defaults)

    def __setstate__(self, state):
        super(ISTA_LIU, self).__setstate__(state)

    def step_(self, model, x, y):
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

            ini_alg(params_with_grad, grads, L, lam, epsilon, 1)

        self.zero_grad()
        y_hat = model(x)
        loss = F.cross_entropy(y_hat, y)
        loss.backward()

        for group in self.param_groups:  # iterate through the parameter groups
            params_with_grad = []
            grads = []

            epsilon = group["epsilon"]
            L = group["L"]
            lam = group["lam"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                params_with_grad.append(p)
                grads.append(p.grad)

            sgd(
                params_with_grad,
                grads,
                [],
                weight_decay=0,
                momentum=0,
                lr=1 / L,
                dampening=0,
                nesterov=False,
                maximize=False
            )
        return


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

            sgd(
                params_with_grad,
                grads,
                [],
                weight_decay=0,
                momentum=0,
                lr=1 / L,
                dampening=0,
                nesterov=False,
                maximize=False
            )
        return
