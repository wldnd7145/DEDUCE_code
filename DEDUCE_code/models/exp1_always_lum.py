# Exp 1: Always LUM (no gradient conflict detection, LUM fires every batch)
from utils.buffer import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from utils.args import *
import torch.nn as nn
import torch
import copy
from utils.gum import GUM
import numpy as np
import torch.nn.functional as F

epsilon = 1E-20

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Exp1: Always LUM')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--beta', type=float, required=True)
    return parser


class Exp1AlwaysLum(ContinualModel):
    NAME = 'exp1_always_lum'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Exp1AlwaysLum, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.temp = copy.deepcopy(self.net).to(self.device)
        self.temp_opt = torch.optim.SGD(self.temp.parameters(), lr=0.01)
        lr = self.args.lr
        self.delta = 0.0001
        self.tau = 0.00001
        self.logsoft = nn.LogSoftmax(dim=1)
        self.checkpoint = {}
        self.fish = {}
        for name, param in self.net.named_parameters():
            self.fish[name] = torch.zeros_like(param).to(self.device)
        self.opt = torch.optim.SGD(self.net.parameters(), lr=lr, weight_decay=0.0001)
        self.phi = 0.00001
        self.gum = GUM(net=self.net, hidden_activation="relu",
                       replacement_rate=self.phi, decay_rate=0.99,
                       util_type="contribution", maturity_threshold=1000,
                       device=self.device)
        self.prev_unlearn_flag = None
        self.times = 0

    def observe(self, inputs, labels, not_aug_inputs, task_id=None, gradients=None, unlearn_flag=None):
        if task_id is not None and task_id != self.prev_unlearn_flag and task_id > 0:
            print(f"task {task_id - 1} unlearn times: {self.times}")
            self.prev_unlearn_flag = task_id
            self.times = 0

        # Always activate LUM when buffer is not empty (no gradient conflict check)
        if task_id > 0 and not self.buffer.is_empty():
            self.lum(inputs=inputs, labels=labels)
            self.times += 1

        self.opt.zero_grad()
        current_features = []
        outputs, features = self.net(inputs, feature_list=current_features)
        loss = self.loss(outputs, labels)
        prev_params = {name: param.clone() for name, param in self.net.named_parameters()}
        loss += self.ewc_loss(prev_params=prev_params, lambda_ewc=0.1)

        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            buf_outputs, feas = self.net(buf_inputs)
            loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
            buf_inputs, buf_labels, _ = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            buf_outputs, feas = self.net(buf_inputs)
            loss += self.args.beta * self.loss(buf_outputs, buf_labels)

        loss.backward()
        self.opt.step()
        self.gum.gen_and_test(current_features, fish=self.fish)
        self.buffer.add_data(examples=not_aug_inputs, labels=labels, logits=outputs.data)
        return loss.item(), features

    def lum(self, inputs, labels):
        self.temp.load_state_dict(self.net.state_dict())
        self.temp.train()
        outputs, feas = self.temp(inputs)
        unlearn_loss = -F.cross_entropy(outputs, labels)
        regularization_loss = 0
        if self.checkpoint:
            for (n1, p1), (n2, p2) in zip(self.checkpoint.items(), self.temp.named_parameters()):
                if n1 == n2:
                    regularization_loss += torch.sum((p2 - p1) ** 2)
        unlearn_loss = unlearn_loss + 0.1 * regularization_loss
        self.temp_opt.zero_grad()
        unlearn_loss.backward()
        self.temp_opt.step()
        for (mn, mp), (tn, tp) in zip(self.net.named_parameters(), self.temp.named_parameters()):
            weight_update = tp - mp
            norm_update = mp.norm() / (weight_update.norm() + epsilon) * weight_update
            identity = torch.ones_like(self.fish[mn])
            with torch.no_grad():
                mp.add_(self.delta * torch.mul(1.0/(identity + 0.001*self.fish[mn]), norm_update + 0.001*torch.randn_like(norm_update)))

    def ewc_loss(self, prev_params, lambda_ewc):
        loss = 0
        for name, param in self.temp.named_parameters():
            if name in prev_params:
                loss += (self.fish[name] * (param - prev_params[name]).pow(2)).sum()
        return lambda_ewc * loss

    def end_task(self, dataset):
        self.temp.load_state_dict(self.net.state_dict())
        fish = {}
        for name, param in self.temp.named_parameters():
            fish[name] = torch.zeros_like(param).to(self.device)
        for j, data in enumerate(dataset.train_loader):
            inputs, labels, _ = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            for ex, lab in zip(inputs, labels):
                self.temp_opt.zero_grad()
                output, feas = self.temp(ex.unsqueeze(0))
                loss = -F.nll_loss(self.logsoft(output), lab.unsqueeze(0), reduction='none')
                exp_cond_prob = torch.mean(torch.exp(loss.detach().clone()))
                loss = torch.mean(loss)
                loss.backward()
                for name, param in self.temp.named_parameters():
                    fish[name] += exp_cond_prob * param.grad ** 2
        for name in fish:
            fish[name] /= (len(dataset.train_loader) * self.args.batch_size)
        for key in self.fish:
            self.fish[key] *= self.tau
            self.fish[key] += fish[key].to(self.device)
        for name, param in self.net.named_parameters():
            self.checkpoint[name] = param.data.clone()
        self.temp_opt.zero_grad()
        return self.fish
