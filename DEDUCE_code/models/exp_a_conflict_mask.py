# Exp A + Conflict Mask: LUM with conflict mask (instead of Fisher) + gradient projection at learning
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
    parser = ArgumentParser(description='Exp A + Conflict Mask')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--beta', type=float, required=True)
    return parser


class ExpAConflictMask(ContinualModel):
    NAME = 'exp_a_conflict_mask'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(ExpAConflictMask, self).__init__(backbone, loss, args, transform)
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
        self.buffer_grad = None
        self.buffer_grad_dict = None
        self.times = 0

    def observe(self, inputs, labels, not_aug_inputs, task_id=None, gradients=None, unlearn_flag=None):
        grad = []
        gradients = []
        if task_id > 0:
            self.buffer_grad, self.buffer_grad_dict = self.cal_buffer(task_id)
        if task_id is not None and task_id != self.prev_unlearn_flag and task_id > 0:
            print(f"task {task_id - 1} unlearn times: {self.times}")
            self.prev_unlearn_flag = task_id
            self.times = 0

        self.opt.zero_grad()
        current_features = []
        outputs, features = self.net(inputs, feature_list=current_features)
        loss = self.loss(outputs, labels)
        grads = torch.autograd.grad(loss, self.net.parameters(), retain_graph=True, create_graph=False)
        if self.buffer_grad is not None:
            for g in grads:
                if g is not None:
                    grad.append(g.view(-1))
            gradients.append(torch.cat(grad))
            gradients = torch.cat(gradients)
            tolerance = 0.0
            grad_norm = torch.norm(gradients, p=2)
            buffer_grad_norm = torch.norm(self.buffer_grad, p=2)
            left = torch.dot(gradients, self.buffer_grad)
            right = tolerance * grad_norm * buffer_grad_norm
            if left <= right:
                new_grad_dict = {}
                for (name, _), g in zip(self.net.named_parameters(), grads):
                    if g is not None:
                        new_grad_dict[name] = g.clone()
                self.lum_conflict_mask(inputs=inputs, labels=labels, new_grad_dict=new_grad_dict)
                self.times += 1

        # LEARN with EWC + gradient projection
        self.opt.zero_grad()
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

        # Gradient projection
        if self.buffer_grad is not None:
            all_grads = torch.cat([p.grad.view(-1) for p in self.net.parameters() if p.grad is not None])
            dot = torch.dot(all_grads, self.buffer_grad)
            if dot < 0:
                proj = dot / (torch.dot(self.buffer_grad, self.buffer_grad) + 1e-8)
                projected = all_grads - proj * self.buffer_grad
                offset = 0
                for p in self.net.parameters():
                    if p.grad is not None:
                        numel = p.numel()
                        p.grad.copy_(projected[offset:offset+numel].view(p.shape))
                        offset += numel

        self.opt.step()
        self.gum.gen_and_test(current_features, fish=self.fish)
        self.buffer.add_data(examples=not_aug_inputs, labels=labels, logits=outputs.data)
        return loss.item(), features

    def lum_conflict_mask(self, inputs, labels, new_grad_dict):
        """LUM with conflict mask instead of Fisher scaling"""
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
            mp_norm = mp.norm()
            wu_norm = weight_update.norm() + epsilon
            norm_update = mp_norm / wu_norm * weight_update

            if self.buffer_grad_dict is not None and mn in self.buffer_grad_dict and mn in new_grad_dict:
                conflict_mask = (self.buffer_grad_dict[mn] * new_grad_dict[mn] < 0).float()
            else:
                conflict_mask = torch.ones_like(mp)

            with torch.no_grad():
                mp.add_(self.delta * conflict_mask * (norm_update + 0.001 * torch.randn_like(norm_update)))

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

    def cal_buffer(self, task_number):
        classes_per_task = 10
        if self.buffer.is_empty():
            return torch.tensor([]), None
        buf_inputs, buf_labels, _ = self.buffer.get_data(self.args.buffer_size, transform=self.transform)
        current_task_label = list(range(classes_per_task * task_number, classes_per_task * (task_number + 1)))
        mask = ~torch.isin(buf_labels, torch.tensor(current_task_label, device=buf_labels.device))
        filtered_inputs = buf_inputs[mask]
        filtered_labels = buf_labels[mask]
        if filtered_inputs.shape[0] == 0:
            return torch.tensor([]), None
        gradients = []
        grad_dicts = []
        for i in torch.arange(classes_per_task):
            task_mask = (filtered_labels // classes_per_task) == i
            task_inputs = filtered_inputs[task_mask]
            task_labels = filtered_labels[task_mask]
            if task_inputs.shape[0] == 0:
                continue
            num_samples = task_inputs.shape[0]
            num_batches = (num_samples + self.args.minibatch_size - 1) // self.args.minibatch_size
            for batch_idx in range(num_batches):
                s = batch_idx * self.args.minibatch_size
                e = min(s + self.args.minibatch_size, num_samples)
                self.opt.zero_grad()
                buf_outputs, feas = self.net(task_inputs[s:e])
                buffer_loss = self.loss(buf_outputs, task_labels[s:e])
                grads = torch.autograd.grad(buffer_loss, self.net.parameters(), retain_graph=False, create_graph=False)
                gradients.append(torch.cat([g.view(-1) for g in grads if g is not None]))
                gd = {}
                for (name, _), g in zip(self.net.named_parameters(), grads):
                    if g is not None:
                        gd[name] = g.clone()
                grad_dicts.append(gd)
        if len(gradients) > 0:
            avg_gradient = torch.stack(gradients).mean(dim=0)
            avg_grad_dict = {}
            for name in grad_dicts[0]:
                avg_grad_dict[name] = torch.stack([gd[name] for gd in grad_dicts]).mean(dim=0)
            return avg_gradient, avg_grad_dict
        return torch.tensor([]), None
