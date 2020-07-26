
import numpy as np
import torch
from torch import nn

class HierNet(nn.Module):
    """Module of hierarchical classifier
    """
    def __init__(self, input_size, nodes):
        super(HierNet, self).__init__()

        self.input_size = input_size
        self.nodes = nodes
        self.num_nodes = len(nodes)

        for i in range(self.num_nodes):
            self.add_module('fc{}'.format(i), nn.Linear(input_size, len(nodes[i].children)))

    def forward(self, x, gate=None, pred=False, thres=0):

        if pred is False:
            # for training
            nout = []
            for i in range(self.num_nodes):
                fc_layers = getattr(self, 'fc{}'.format(i))
                nout.append(fc_layers(x))

            outs = []
            out_masks = []
            # root node (no dependency to other nodes)
            cw = torch.from_numpy(self.nodes[0].codeword).float().to(nout[0].device)
            outs.append(torch.matmul(nout[0], cw) * gate[:, 0].view(-1, 1))
            # other internal nodes
            for i in range(1, self.num_nodes):
                cw = torch.from_numpy(self.nodes[i].codeword).float().to(nout[i].device)
                cond = self.nodes[i].cond
                cond_gate = torch.ones([x.size(0), 1]).to(nout[i].device)
                while cond:
                    parent, _ = cond
                    cond_gate = torch.mul(cond_gate, gate[:, parent].view(-1, 1))
                    cond = self.nodes[parent].cond
                outs.append(torch.matmul(nout[i], cw) * cond_gate * gate[:, i].view(-1, 1))
                mask = torch.clamp(torch.from_numpy(self.nodes[i].mask).float().to(nout[i].device), 1e-17, 1)
                out_masks.append(torch.log(mask) * (1 - gate[:, i].view(-1, 1)))

            self.output = torch.clamp(torch.sum(torch.stack(outs), 0), -1e17, 10)
            out_mask = torch.eq(torch.sum(torch.stack(out_masks), 0), 0).float()
            self.sfmx_base = torch.sum(torch.exp(self.output) * out_mask, 1)

            return self.output, nout, self.sfmx_base

        else:
            # for testing
            if gate is None:
                gate = torch.zeros([x.size(0), self.num_nodes]).to(x.device)
                nout = []
                for i in range(self.num_nodes):
                    fc_layers = getattr(self, 'fc{}'.format(i))
                    nout.append(fc_layers(x))
                    cf = torch.max(torch.softmax(nout[i], dim=1), dim=1)[0]
                    gate[:, i] = torch.ge(cf, thres).float()
            else:
                nout = []
                for i in range(self.num_nodes):
                    fc_layers = getattr(self, 'fc{}'.format(i))
                    nout.append(fc_layers(x))

            outs = []
            # root node (no dependency to other nodes)
            cw = torch.from_numpy(self.nodes[0].codeword).float().to(nout[0].device)
            outs.append(torch.matmul(nout[0], cw) * gate[:, 0].view(-1, 1))
            # other internal nodes
            for i in range(1, self.num_nodes):
                cw = torch.from_numpy(self.nodes[i].codeword).float().to(nout[i].device)
                cond = self.nodes[i].cond
                cond_gate = torch.ones([x.size(0), 1]).to(nout[i].device)
                while cond:
                    parent, _ = cond
                    cond_gate = torch.mul(cond_gate, gate[:, parent].view(-1, 1))
                    cond = self.nodes[parent].cond
                outs.append(torch.matmul(nout[i], cw) * cond_gate * gate[:, i].view(-1, 1))

            self.output = torch.sum(torch.stack(outs), 0)

            return self.output, nout


def hiernet(**kwargs):
    model = HierNet(**kwargs)
    return model

