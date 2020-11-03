import os
import shutil
from datetime import datetime
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

EQUIVARIANCES = ['translations']
ACTIVATIONS = ['relu', 'maxpool', 'upsample']


class Conv1d_pad(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(Conv1d_pad, self).__init__(in_channels, out_channels,
                                         kernel_size, bias=False)
        self.equivaricance = 'translations'
        self.padding = (self.kernel_size[0]//2)

    def __repr__(self):
        return "{}\nkernel size{}".format("Conv1d", self.kernel_size)


class Conv2d_pad(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(Conv2d_pad, self).__init__(in_channels, out_channels,
                                         kernel_size, bias=False)
        self.equivaricance = 'translations'
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)

    def __repr__(self):
        return "{}\nkernel size{}".format("Conv2d", self.kernel_size)


class LinearNoBias(nn.Linear):
    def __init__(self, in_features, out_features):
        super(LinearNoBias, self).__init__(in_features, out_features,
                                           bias=False)
        self.equivariance = "identity"

    def __repr__(self):
        return "Linear, no bias"


def morphism_func(equivariance, dimension=2):
    if dimension == 2:
        translations = Conv2d_pad
    elif dimension == 1:
        translations = Conv1d_pad
    else:
        msg = "Dimennsion {} is not supported, use  1 or 2".format(dimension)
        raise NotImplementedError(msg)
    return {'translations': translations,
            'identity': LinearNoBias}[equivariance]


def activation_func(activation, dimension=2):
    if dimension == 2:
        maxpool = nn.MaxPool2d(2)
        upsample = nn.UpsamplingBilinear2d(scale_factor=2)
    elif dimension == 1:
        maxpool = nn.MaxPool1d(2)
        upsample = nn.Upsample(scale_factor=2)
    else:
        msg = "Dimennsion {} is not supported, use  1 or 2".format(dimension)
        raise NotImplementedError(msg)
    return nn.ModuleDict([['relu', nn.ReLU6(inplace=False)],
                          ['maxpool', maxpool],
                          ['upsample', upsample],
                          ['linear', nn.Identity()]])[activation]


def l2_norm(param):
    size_reg = param.numel()
    return torch.sqrt(torch.sum(torch.pow(param, 2)) / size_reg)


def reg_loss(output, target, model, loss_func, reg_coeff=1, n_incoming=3):
    spaces = model.spaces[model.number_of_input_spaces:]
    spaces.extend(model.output_spaces)

    for i, space in enumerate(spaces):
        if i == 0:
            reg = 0
        if not space.pruned:
            if i <= model.number_of_spaces:
                coeff = len(space.incoming) - n_incoming
            else:
                coeff = 1
            if coeff < 0:
                coeff = 0

            for param in space.parameters():
                reg += coeff * l2_norm(param)

    l1 = loss_func(output, target)
    l2 = reg_coeff * reg
    return l1, l2


class Morphism(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=None, dimension=2,
                 equivariance='translations', prunable=True, origin=None,
                 destination=None):
        super(Morphism, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.equivariance = equivariance
        params = self.get_equivariance_params()
        self.model = nn.Sequential(morphism_func(equivariance,
                                                 dimension=dimension)(*params))
        self.volume_buffer = None
        self.prunable = prunable
        self.origin = origin
        self.destination = destination
        self._pruned = False

    def get_equivariance_params(self):
        if self.equivariance == "identity":
            return [self.in_ch, self.out_ch]
        else:
            return [self.in_ch, self.out_ch, self.kernel_size]

    @property
    def pruned(self):
        return self._pruned

    def pruning_condition(self, tol):
        return torch.lt(self.weight_l2_norm(), tol) if not self.pruned else True

    def prune(self, tol):
        if self.prunable and self.pruning_condition(tol):
            self._pruned = True

    def weight_l1_norm(self):
        for i, param in enumerate(self.parameters()):
            if i == 0:
                norm = torch.sum(torch.abs(param))
            else:
                norm += torch.sum(torch.abs(param))

        return norm

    def weight_l2_norm(self):
        for i, param in enumerate(self.parameters()):
            if i == 0:
                norm = l2_norm(param)
            else:
                norm += l2_norm(param)

        return norm

    def forward(self, inputs):
        if self.equivariance == "identity":
            inputs = inputs.view(inputs.size(0), -1)
        return self.model(inputs)


class Space(nn.Module):
    def __init__(self, in_size, out_size, num_channels, dimension=2,
                 incoming_morphisms=[], activation="linear", index=None,
                 prunable=True, is_input=False, is_output=False):
        super(Space, self).__init__()
        self.incoming = nn.ModuleList(incoming_morphisms)
        self.in_size = in_size
        self.out_size = out_size
        self.num_channels = num_channels
        self.dimension = dimension
        if self.dimension == 1:
            self.size = (self.num_channels, self.out_size)
        else:
            self.size = (self.num_channels,) + tuple(self.out_size)
        self.activation_name = activation
        self.activation = activation_func(activation, dimension=dimension)
        self.index = index
        self._depth = 0
        self._pruned = False
        self.prunable = prunable
        self.is_output = is_output
        self.is_input = is_input

    @property
    def pruned(self):
        if not self.prunable:
            return torch.BoolTensor([False])
        pruned_incoming = torch.BoolTensor([m.pruned for m in self.incoming])
        return pruned_incoming.all()

    @property
    def depth(self):
        return self._depth

    def add_incoming(self, morphisms):
        self.incoming += morphisms

    def forward(self, spaces):
        out = None

        for i, morphism in enumerate(self.incoming):
            space = spaces[morphism.origin]
            if not morphism.pruned:
                if i == 0 or out is None:
                    out = morphism.forward(space.volume_buffer)
                else:
                    out += morphism.forward(space.volume_buffer)

        out = self.activation(out)
        self.volume_buffer = out
        return out

    def __repr__(self):
        return "{}\n{}".format(self.index, self.activation_name)


class BestModelSaver:
    def __init__(self, path):
        self.loss = 1e+6
        self.check_path(path)
        self.ckpt_folder = path
        self.params_path = os.path.join(path, "params.npy")
        self.path = os.path.join(path, 'checkpoint.pth.tar')
        self.best_path = os.path.join(path, 'model_best.pth.tar')

    @staticmethod
    def check_path(path):
        if not os.path.isdir(path):
            os.makedirs(path)

    def is_best(self, new_loss):
        return self.loss > new_loss

    def save(self, model, optimizer, epoch, loss, acc):
        print(epoch, '-'*8)
        if epoch == 1:
            params_dict = {
                'input_size': model.input_size,
                'number_of_spaces': model.number_of_spaces,
                'dimension': model.dimension,
                'num_channels': model.num_channels,
                'activations': model.activations,
                'equivariances': model.equivariances,
                'kernel_size': model.kernel_size, 'tol': model.tol,
                'number_of_input_spaces': model.number_of_input_spaces,
                'number_of_classes': model.number_of_classes,
                'limit_image_upsample': model.limit_image_upsample,
                'classifier': model.classifier, 'prune': model.prune_it}
            np.save(self.params_path, params_dict)
        state = {
            'epoch': epoch,
            'loss': loss,
            'state_dict': model.state_dict(),
            'best_acc1': acc,
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, self.path)
        if self.is_best(loss):
            self.loss = loss
            shutil.copyfile(self.path, self.best_path)


class BestModelLoader:
    def __init__(self, path_to_ckpt_folder, model_class=None):
        assert os.path.isdir(path_to_ckpt_folder), ("provide a valid ckpt dir")
        self.path = path_to_ckpt_folder
        self.model_class = model_class
        self.model = self.load()

    def load(self):
        self.params_dict = np.load(os.path.join(self.path, 'params.npy'),
                                   allow_pickle=True).item()
        self.path = os.path.join(self.path, 'model_best.pth.tar')
        state_dict = torch.load(self.path)
        model = self.model_class(**self.params_dict)
        model.load_state_dict(state_dict['state_dict'])
        return model


def train(model, device, train_loader, optimizer, epoch, loss_func=F.nll_loss,
          loss_inputs=None, ret=False):
    steps = 10
    correct = 0
    n_total = 0
    epoch_accs = []
    epoch_loss = []
    epoch_reg_loss = []
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        if model.dimension == 1:
            data = data.view(data.size(0), 1, -1)
        optimizer.zero_grad()
        output = model([data])
        l1, l2 = loss_func(output, target, *loss_inputs)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        n_total += data.shape[0]
        (l1 + l2).backward()
        optimizer.step()
        epoch_accs.append(100. * correct / n_total)
        epoch_loss.append(l1.item())
        epoch_reg_loss.append(l2.item())
        if batch_idx % steps == 0 and batch_idx != 0:
            print('Train Epoch {}: [{}/{} ({:.0f}%)]\tLoss: {:.3f}\tAcc {:.3f}%'.format(
                      epoch, batch_idx * len(data), len(train_loader.dataset),
                      100. * batch_idx / len(train_loader), (l1 + l2).item(),
                      100. * correct / n_total))
            print("main loss: {:.3f}, ret loss: {:.3f}".format(l1.item(),
                                                               l2.item()))
    if ret:
        return (np.mean(epoch_loss), np.mean(epoch_reg_loss),
                np.mean(epoch_accs))


def test(model, device, test_loader, loss_func=F.nll_loss):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if model.dimension == 1:
                data = data.view(data.size(0), 1, -1)
            output = model([data])
            test_loss += loss_func(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set:'
          +' Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), test_acc))
    return test_loss, test_acc


def generate_timestamp():
    return datetime.now().isoformat()[:-7].replace("T","-").replace(":","-")


def get_spaced_colors(n, norm=False, black=True, cmap='jet'):
    rgb_tuples = cm.get_cmap(cmap)
    if norm:
        colors = [rgb_tuples(i / n) for i in range(n)]
    else:
        rgb_array = np.asarray([rgb_tuples(i / n) for i in range(n)])
        brg_array = np.zeros(rgb_array.shape)
        brg_array[:, 0] = rgb_array[:, 2]
        brg_array[:, 1] = rgb_array[:, 1]
        brg_array[:, 2] = rgb_array[:, 0]
        colors = [tuple(brg_array[i, :] * 256) for i in range(n)]
    if black:
        black = (0., 0., 0.)
        colors.insert(0, black)
    return colors


def compute_out_shape(size, layer):
    if is_conv_like(layer):
        h_out, w_out = conv_out_shape(size, layer)
    elif is_upsampling:
        h_out, w_out = upsampling_out_shape(size, layer)
    else:
        print(layer)
        message = "Please implement a function to return out shape"
        raise NotImplementedError(message)
    return h_out, w_out


def conv_out_shape(size, conv_op):
    from math import floor
    p = get_conv_like_attrs(conv_op.padding)
    d = get_conv_like_attrs(conv_op.dilation)
    k = get_conv_like_attrs(conv_op.kernel_size)
    s = get_conv_like_attrs(conv_op.stride)
    h = floor(((size[0] + (2 * p[0]) - d[0] * (k[0] - 1) - 1) / s[0]) + 1)
    w = floor(((size[1] + (2 * p[1]) - d[1] * (k[1] - 1) - 1) / s[1]) + 1)
    return h, w


def upsampling_out_shape(size, conv_op):
    if conv_op.scale_factor is not None:
        return size[0] * conv_op.scale_factor, size[1] * conv_op.scale_factor
    elif conv_op.size is not None:
        s = get_conv_like_attrs(conv_op.size)
        return size[0] * s[0], size[1] * s[1]


def get_conv_like_attrs(attr):
    if isinstance(attr, int):
        attr = [attr, attr]
    elif len(attr) == 1:
        attr = [attr[0], attr[0]]
    return attr


def is_upsampling(layer):
    return (isinstance(layer, nn.Upsample)
            or isinstance(layer, nn.UpsamplingNearest2d)
            or isinstance(layer, nn.UpsamplingBilinear2d))


def is_conv_like(layer):
    return (isinstance(layer, nn.Conv2d)
            or isinstance(layer, nn.MaxPool2d))


def get_graph(hm):
    graph = nx.DiGraph()
    [graph.add_node(i, color=get_node_color(hm.get_space_by_index(i)))
     for i in range(len(hm.spaces) + len(hm.output_spaces))]
    [graph.add_edge(m.origin, m.destination, model=m.model,
                    color=get_edge_color(m))
     for s in hm.spaces[1:] for m in s.incoming if not m.pruned]
    [graph.add_edge(m.origin, m.destination, model=m.model,
                    color=get_edge_color(m))
     for s in hm.output_spaces for m in s.incoming if not m.pruned]
    return graph


def get_node_color(space, colors=["w", "g", "y"]):
    if space.is_input:
        return colors[0]
    elif space.is_output:
        return colors[1]
    else:
        return colors[2]


def get_edge_color(morphism, colors=["k", "b", "r"]):
    if morphism.equivariance == "identity":
        return colors[0]
    elif morphism.equivariance == "translations":
        return colors[1]
    else:
        return colors[2]


def visualise_graph(hm, ax=None):
    graph = get_graph(hm)
    if ax is None:
        f, ax = plt.subplots()
    pos = nx.circular_layout(graph)
    n_colors = [n[1] for n in graph.nodes.data('color')]
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=n_colors)
    labels = {s.index: s for s in hm.spaces}

    for s in hm.output_spaces:
        labels[s.index] = s

    nx.draw_networkx_labels(graph, pos, ax=ax, labels=labels)
    e_colors = [e[2] for e in graph.edges.data('color')]
    nx.draw_networkx_edges(graph, pos, arrows=True, ax=ax,
                           edge_color=e_colors)
    lines = {"translations": Line2D([0], [0], color='b', lw=4),
             "identity": Line2D([0], [0], color='k', lw=4)}
    custom_lines = [lines[e] for e in hm.equivariances + ["identity"]]
    ax.legend(custom_lines, hm.equivariances + ["identity"])
