# Exp C: Replace GUM neuron reinitialization with stochastic dropout
# Instead of permanently reinitializing low-contribution neurons,
# apply dropout p=0.5 to bottom phi fraction of neurons each batch
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
    parser = ArgumentParser(description='Exp C: Dropout GUM')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--beta', type=float, required=True)
    return parser


class DropoutGUM:
    """Replace GUM's neuron reinitialization with stochastic dropout on low-contribution neurons"""
    def __init__(self, net, phi=0.00001, decay_rate=0.99, maturity_threshold=1000, device='cpu'):
        self.net = net
        self.phi = phi
        self.decay_rate = decay_rate
        self.maturity_threshold = maturity_threshold
        self.device = device

        # Collect conv layers (exclude 1x1 shortcut) and linear
        self.weight_layers = []
        self._get_weight_layers(net)
        self.num_hidden = len(self.weight_layers) - 1

        self.util = []
        self.ages = []
        self.mean_feature_mag = []
        for i in range(self.num_hidden):
            n = self.weight_layers[i].out_channels if hasattr(self.weight_layers[i], 'out_channels') else self.weight_layers[i].out_features
            self.util.append(torch.zeros(n, device=device))
            self.ages.append(torch.zeros(n, device=device))
            self.mean_feature_mag.append(torch.zeros(n, device=device))

        # Dropout masks stored per layer
        self.dropout_masks = [None] * self.num_hidden

    def _get_weight_layers(self, module):
        if isinstance(module, nn.Conv2d) and module.kernel_size != (1, 1):
            self.weight_layers.append(module)
        elif isinstance(module, nn.Linear):
            self.weight_layers.append(module)
        else:
            if isinstance(module, nn.Sequential):
                for m in module.children():
                    if isinstance(m, nn.BatchNorm2d):
                        continue
                    self._get_weight_layers(m)
            else:
                for m in module.children():
                    if hasattr(module, 'downsample') and module.downsample == m:
                        continue
                    self._get_weight_layers(m)

    def gen_and_test(self, features, fish=None):
        if not isinstance(features, list) or len(features) == 0:
            return

        from torch.nn.functional import sigmoid

        fish_conv = {key: value for key, value in fish.items() if "conv" in key or "fc.weight" in key} if fish else {}

        for i in range(self.num_hidden):
            if i >= len(features):
                break
            self.ages[i] += 1

            # Update mean feature magnitude
            with torch.no_grad():
                if features[i].dim() == 2:
                    self.mean_feature_mag[i] += (1 - self.decay_rate) * features[i].abs().mean(dim=0)
                elif features[i].dim() == 4:
                    self.mean_feature_mag[i] += (1 - self.decay_rate) * features[i].abs().mean(dim=(0, 2, 3))

            # Check eligible neurons
            eligible = torch.where(self.ages[i] > self.maturity_threshold)[0]
            if eligible.shape[0] == 0:
                continue

            # Compute contribution (same as original GUM)
            with torch.no_grad():
                next_layer = self.weight_layers[i + 1]
                if isinstance(next_layer, nn.Linear):
                    output_weight_mag = next_layer.weight.data.abs().mean(dim=0)
                    if len(fish_conv) > i + 1:
                        layer_name = list(fish_conv.keys())[i + 1]
                        fisher_info = fish_conv[layer_name]
                        unit_fish = fisher_info.data.abs().mean(dim=0)
                        I_norm = (unit_fish - unit_fish.min()) / (unit_fish.max() - unit_fish.min() + 1e-8)
                        gate = sigmoid(10 * I_norm)
                    else:
                        gate = torch.ones_like(output_weight_mag)
                elif isinstance(next_layer, nn.Conv2d):
                    output_weight_mag = next_layer.weight.data.abs().mean(dim=(0, 2, 3))
                    if len(fish_conv) > i + 1:
                        layer_name = list(fish_conv.keys())[i + 1]
                        fisher_info = fish_conv[layer_name]
                        unit_fish = fisher_info.data.abs().mean(dim=(0, 2, 3))
                        I_norm = (unit_fish - unit_fish.min()) / (unit_fish.max() - unit_fish.min() + 1e-8)
                        gate = sigmoid(10 * I_norm)
                    else:
                        gate = torch.ones_like(output_weight_mag)

                self.util[i] = output_weight_mag * self.mean_feature_mag[i] * gate

            # Instead of reinitializing: apply dropout to bottom phi neurons
            n_neurons = eligible.shape[0]
            n_to_dropout = max(1, int(self.phi * n_neurons))
            if n_to_dropout > 0:
                # Find lowest contribution neurons among eligible
                bottom_indices = eligible[torch.topk(-self.util[i][eligible], min(n_to_dropout, eligible.shape[0]))[1]]
                # Apply dropout mask p=0.5 to these neurons' outgoing weights
                with torch.no_grad():
                    next_layer = self.weight_layers[i + 1]
                    mask = torch.bernoulli(torch.full((len(bottom_indices),), 0.5, device=self.device)).bool()
                    neurons_to_zero = bottom_indices[mask]
                    if len(neurons_to_zero) > 0:
                        if isinstance(next_layer, nn.Conv2d):
                            if neurons_to_zero.max() < next_layer.weight.data.shape[1]:
                                next_layer.weight.data[:, neurons_to_zero] *= 0.5
                        elif isinstance(next_layer, nn.Linear):
                            if neurons_to_zero.max() < next_layer.weight.data.shape[1]:
                                next_layer.weight.data[:, neurons_to_zero] *= 0.5


class ExpCDropoutGum(ContinualModel):
    NAME = 'exp_c_dropout_gum'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(ExpCDropoutGum, self).__init__(backbone, loss, args, transform)
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
        # Dropout GUM instead of original GUM
        self.gum = DropoutGUM(net=self.net, phi=0.00001, decay_rate=0.99,
                              maturity_threshold=1000, device=self.device)
        self.prev_unlearn_flag = None
        self.buffer_grad = None
        self.times = 0

    def observe(self, inputs, labels, not_aug_inputs, task_id=None, gradients=None, unlearn_flag=None):
        grad = []
        gradients = []
        if task_id > 0:
            self.buffer_grad = self.cal_buffer(task_id)
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
                self.lum(inputs=inputs, labels=labels)
                self.times += 1

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

    def cal_buffer(self, task_number):
        classes_per_task = 10
        if self.buffer.is_empty():
            return torch.tensor([])
        buf_inputs, buf_labels, _ = self.buffer.get_data(self.args.buffer_size, transform=self.transform)
        current_task_label = list(range(classes_per_task * task_number, classes_per_task * (task_number + 1)))
        mask = ~torch.isin(buf_labels, torch.tensor(current_task_label, device=buf_labels.device))
        filtered_inputs = buf_inputs[mask]
        filtered_labels = buf_labels[mask]
        if filtered_inputs.shape[0] == 0:
            return torch.tensor([])
        gradients = []
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
        if len(gradients) > 0:
            return torch.stack(gradients).mean(dim=0)
        return torch.tensor([])
