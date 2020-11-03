from torch import nn
import torch.nn.functional as F
import numpy as np
from hypergraph_machines.utils import Space, Morphism
from hypergraph_machines.utils import ACTIVATIONS
from hypergraph_machines.utils import get_graph
EQUIVARIANCES = ["translations"]


class HypergraphMachine(nn.Module):
    def __init__(self, input_size=None, number_of_spaces=None, num_channels=4,
                 limit_image_upsample=3, number_of_input_spaces=1,
                 activations=ACTIVATIONS, equivariances=EQUIVARIANCES,
                 tol=1e-6, classifier=None, number_of_classes=None,
                 prune=True, dimension=2, kernel_size=3):
        super(HypergraphMachine, self).__init__()
        self.number_of_spaces = number_of_spaces
        self.input_size = input_size  # (ch, h, w)
        self.dimension = dimension
        self.num_channels = num_channels
        self.activations = activations
        self.equivariances = equivariances
        self.kernel_size = kernel_size
        self.tol = tol
        self.input_channels = input_size[0]
        if dimension == 2:
            self.sizes = [input_size[1:]]
        else:
            self.input_size_prod = int(np.prod(input_size[1:]))
            self.sizes = [int(np.prod(input_size[1:]))]
        self.number_of_input_spaces = number_of_input_spaces
        self.number_of_classes = number_of_classes
        self.limit_image_upsample = limit_image_upsample
        self.classifier = classifier
        self.prune_it = prune
        self.spaces = nn.ModuleList([])
        self.build_spaces()

    def resample_activation(self, out_size):
        # print("input size ", self.input_size)
        if out_size[0] > self.input_size[1] * self.limit_image_upsample:
            return True
        elif out_size[0] < self.input_size[1] / self.limit_image_upsample:
            return True
        else:
            return False

    def resample_activation1d(self, out_size):
        # print("input size ", self.input_size)
        if out_size > self.input_size_prod * self.limit_image_upsample:
            return True
        elif out_size < self.input_size_prod / self.limit_image_upsample:
            return True
        else:
            return False

    def sample_activation(self):
        res = True

        while res:
            a = self.random_choice(self.activations)
            in_size = self.random_choice(self.sizes)
            out_size = (np.asarray(in_size)
                        * self.get_size_coeff(a)).astype(np.int)

            if self.dimension == 1:
                res = self.resample_activation1d(out_size)
            else:
                res = self.resample_activation(out_size)

        return a, in_size, out_size

    def sample_equivariance(self):
        return self.random_choice(self.equivariances)

    @staticmethod
    def random_choice(list):
        array = np.asarray(list)
        ind = np.random.choice(array.shape[0])
        return array[ind]

    @staticmethod
    def get_size_coeff(activation):
        if activation == "maxpool":
            return .5
        elif activation == "upsample":
            return 2
        else:
            return 1

    def get_input_spaces(self):
        for i in range(self.number_of_input_spaces):
            self.spaces.append(Space(self.sizes[0], self.sizes[0], index=i,
                                     num_channels=self.num_channels,
                                     prunable=False, is_input=True,
                                     dimension=self.dimension))

        [setattr(s, "_depth", 0) for s in self.spaces]

    def get_space(self, index):
        a, in_size, out_size = self.sample_activation()
        self.sizes.extend([in_size, out_size])
        space = Space(in_size, out_size, incoming_morphisms=[],
                      activation=a, index=index,
                      num_channels=self.num_channels,
                      dimension=self.dimension)
        self.set_incoming_morphisms(space)
        return space

    def set_spaces(self):
        num_inputs = self.number_of_input_spaces

        for i in range(num_inputs, self.number_of_spaces + num_inputs):
            space = self.get_space(i)
            self.spaces.append(space)

    def build_spaces(self):
        self.get_input_spaces()
        self.set_spaces()
        if self.number_of_classes is not None:
            self.build_classifiers()
        self.graph = get_graph(self)

    def init_classifier(self, space):
        if self.dimension == 1:
            out_shape = (space.num_channels, space.out_size)
        else:
            out_shape = ((space.num_channels,) + tuple(space.out_size))
        if self.number_of_classes is not None:
            out1 = self.number_of_classes
            clf = self.get_output_space(np.prod(out_shape), out1, space.index,
                                        space.index)
        return clf

    def get_output_morphism(self, in_features, out_features, i, j):
        return Morphism(in_features, out_features, equivariance='identity',
                        origin=i, destination=j)

    def get_output_space(self, in_features, out_features, i, j):
        index = j + self.number_of_spaces
        s = Space(in_features, out_features, 1, activation="relu",
                  index=index, is_output=True, dimension=1)
        s.add_incoming([self.get_output_morphism(in_features, out_features, i,
                                                 index)])
        self.set_depth(s)
        return s

    def build_classifiers(self):
        self.output_spaces = nn.ModuleList([])

        if self.number_of_classes is not None:
            for i, s in enumerate(self.spaces[self.number_of_input_spaces:]):
                clf = self.init_classifier(s)
                self.output_spaces.append(clf)

    def set_incoming_morphisms(self, space):
        inc = []

        for j, s in enumerate(self.spaces):
            if np.all(s.out_size == space.in_size):
                if j < self.number_of_input_spaces:
                    num_in_ch = self.input_channels
                    prunable = False
                else:
                    num_in_ch = self.num_channels
                    prunable = True
                equivariance = self.sample_equivariance()
                m = Morphism(num_in_ch, self.num_channels, self.kernel_size,
                             equivariance=equivariance, prunable=prunable,
                             origin=j, destination=space.index,
                             dimension=self.dimension)
                inc.append(m)

        space.add_incoming(inc)
        self.set_depth(space)

    def get_outgoing(self, space):
        spaces = [m for s in self.spaces for m in s.incoming
                  if m.origin == space.index]
        output = [m for s in self.output_spaces for m in s.incoming
                  if m.origin == space.index]
        return spaces + output

    def set_depth(self, space):
        depths = [self.spaces[morphism.origin].depth for morphism in
                  space.incoming]
        depth = max(depths)
        setattr(space, "_depth", depth + 1)

    def forward(self, inputs):
        x, y = inputs, None

        for i, space in enumerate(self.spaces[:self.number_of_spaces]):
            if i < self.number_of_input_spaces:
                space.volume_buffer = x[i]
            elif not space.pruned and i >= self.number_of_input_spaces:
                x = space.forward(self.spaces)
                ind = space.index - self.number_of_input_spaces
                if not self.output_spaces[ind].pruned:
                    if y is None:
                        y = self.output_spaces[ind].forward(self.spaces)
                    else:
                        y += self.output_spaces[ind].forward(self.spaces)

        out = F.log_softmax(y, dim=1)
        if self.prune_it:
            self.set_depth_and_prune(self.spaces[self.number_of_input_spaces:])
            self.set_depth_and_prune(self.output_spaces)
        return out

    def set_depth_and_prune(self, space_list):
        for space in space_list:
            self.set_depth(space)
            self.prune(space)

    def prune(self, space):
        [m.prune(self.tol) for m in space.incoming if not m.pruned]
        outgoing = self.get_outgoing(space)
        if (len(space.incoming) == 0 or space.pruned) and not space.is_input:
            [setattr(m, "_pruned", True) for m in outgoing]
        elif (len(outgoing) == 0
              or len([m for m in outgoing if not m.pruned]) == 0):
            [setattr(m, "_pruned", True) for m in space.incoming
             if not space.is_output]

    def get_space_by_index(self, index):
        s1 = [s for s in self.spaces if s.index == index]
        s2 = [s for s in self.output_spaces if s.index == index]
        return (s1+s2)[0]
