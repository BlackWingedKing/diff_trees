"""
    This code parses a tree and returns a pytorch model
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DTModel(nn.Module):
    """
        Adaptively build the tree from the parser

        The idea is each of tree node is a neuron
        and activation with sigmoid(ai.xi - ti)
        and weights are not differentiable (ai is an indicator)

        inputs:
        nclasses: num_classes for classification
        root: the sklearn dt tree's root node
    """
    def __init__(self, nclasses, root):
        super(DTModel, self).__init__()
        # base attributes
        self.root  = root
        self.features = root.feature
        self.threshold = root.threshold
        self.values = root.value
        self.children_left = root.children_left
        self.children_right = root.children_right

        self.nclasses = nclasses
        self.nnodes = root.node_count

        # create bias_list parameter check it registers not needed parameters too...
        self.bias = nn.ParameterDict()

        # register the differentiable biases into the dict
        for i in range(self.nnodes):
            if(self.features[i]>=0):
                # if the feature >= 0 i.e the non-leaf nodes
                _bias = nn.Parameter(torch.ones(1)*self.threshold[i])
                self.bias[str(i)] = _bias

    def _approx_activation_by_index(self, feat_input, feat_index, threshold):
        activation = torch.sigmoid(feat_input[:, feat_index] - threshold)
        return 1.0 - activation, activation

    def _split_approx(self, node, feat_input, feat_index, threshold):
        if (node is None):
            node = 1.0
        l_n, r_n = self._approx_activation_by_index(feat_input, feat_index, threshold)
        return node*l_n, node*r_n

    def forward(self, x):
        # a condition if people pass input in a funny way instead of Bxinput_size 
        # this condition check is till unclear and needs to be discussed with amit and..
        flag = False
        if(len(x.shape) == 1):
            x = x.view(1, -1)
            flag = True
        
        nodes = [None]*(self.nnodes)
        leaf_nodes = [[] for _ in range(self.nclasses)]

        for i in range(self.nnodes):
            cur_node = nodes[i]
            if (self.children_left[i] != self.children_right[i]):
                l_n, r_n = self._split_approx(cur_node, x, self.features[i], self.bias[str(i)])
                nodes[self.children_left[i]] = l_n
                nodes[self.children_right[i]] = r_n            
            else:
                max_class = np.argmax(self.values[i])
                leaf_nodes[max_class].append(cur_node)

        # any other special cases??
        if(self.nnodes > 1):
            out_l = [sum(leaf_nodes[c_i]) for c_i in range(self.nclasses)]
            # print(out_l)
            p = torch.stack(out_l, dim=-1)

        else:
            # tree built with a single node
            exit("tree built with single node, this can't be converted to a proper model yet")

        # condition when we only have 2 classes so, we only need a single neuron output 
        # instead of 1-p, p

        if (self.nclasses == 2):
            if(flag):
                return p[:, 1]
            else:
                return p[:, 1].view(-1, 1)
        else:
            return p

def parse_tree(tree):
    """
        parse the tree return the torch model
    """
    nclasses = len(tree.classes_)
    return DTModel(nclasses, tree.tree_)
