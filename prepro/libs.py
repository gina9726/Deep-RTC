
import numpy as np
import os


class TreeNode():

    def __init__(self, name, path, depth, node_id, child_idx=-1, parent=None):
        self.name = name
        self.path = path
        self.depth = depth
        self.node_id = node_id
        self.children = {}
        self.child_idx = child_idx
        self.parent = parent
        self.codeword = None
        self.cond = None
        self.children_unid = None
        self.mask = None

    def add_child(self, child):
        self.children[len(self.children)] = child

    def init_codeword(self, cw_size):
        self.codeword = np.zeros([cw_size])

    def set_codeword(self, idx):
        self.codeword[idx] = 1

    def set_cond(self, parent_idx):
        self.cond = [parent_idx, self.child_idx]

    def __str__(self):
        attr = 'name={}, node_id={}, depth={}, children={}'.format(
                    self.name, self.node_id, self.depth,
                    ','.join([chd for chd in self.children.values()])
                )
        return  attr

    def copy(self):
        new_node = TreeNode(self.name, self.path, self.depth, self.node_id, self.child_idx, self.parent)
        new_node.children = self.children.copy()
        if self.cond:
            new_node.cond = self.cond.copy()
        return new_node


class Tree():

    def __init__(self, data_root):
        self.root = TreeNode('root', data_root, 0, 0)
        self.depth = 0
        self.nodes = {'root': self.root}
        self._buildTree(self.root)
        self.used_nodes = {}
        self.leaf_nodes = {}

    def _buildTree(self, root, depth=0):

        for chd in os.listdir(root.path):
            chd_path = os.path.join(root.path, chd)

            if os.path.isdir(chd_path):
                assert chd not in self.nodes
                child_idx = len(root.children)
                root.add_child(chd)
                node_id = len(self.nodes)
                child = TreeNode(chd, chd_path, depth + 1, node_id, child_idx, root.name)
                self.nodes[chd] = child

                self._buildTree(child, depth + 1)

        self.depth = max(self.depth, depth)

    def show(self, node_name='root', root_depth=-1, max_depth=np.Inf):

        root = self.nodes.get(node_name, None)
        if not root:
            raise ValueError('{} is not in the tree'.format(node_name))

        if root_depth == -1:
            print(root.name)
            root_depth = root.depth
            max_depth = min(self.depth, max_depth)

        if root.depth - root_depth < max_depth:
            for chd in root.children.values():
                child = self.nodes[chd]
                print('--' * (child.depth - root_depth), end='')
                print(child.name)
                self.show(chd, root_depth, max_depth)

    def gen_codeword(self, max_depth=np.Inf):

        if max_depth == np.Inf:
            leaf_nodes = sorted([x.name for x in self.nodes.values() if len(x.children) == 0])
        elif max_depth <= self.depth:
            leaf_nodes = sorted([x.name for x in self.nodes.values() if x.depth == max_depth])
        else:
            raise ValueError('max_depth should be equal or smaller than {}'.format(self.depth))

        used_nodes = [x for x in self.nodes.values() if x.depth < max_depth and x.name not in leaf_nodes]
        used_nodes = sorted(used_nodes, key=lambda x: x.node_id)
        self.used_nodes = dict(enumerate([x.name for x in used_nodes]))

        self.leaf_nodes = dict(enumerate(leaf_nodes))
        node_list = [x for x in self.used_nodes.values()] + [x for x in self.leaf_nodes.values()]
        for n in node_list:
            node = self.nodes.get(n)
            node.init_codeword(len(self.leaf_nodes))

        for idx, n in self.leaf_nodes.items():
            node = self.nodes.get(n)
            node.set_codeword(idx)
            parent = self.nodes.get(node.parent)
            # reverse traversal
            while parent.name != 'root':
                parent.set_codeword(idx)
                parent = self.nodes.get(parent.parent)
            parent.set_codeword(idx)

    def gen_rel_path(self):

        name2Id = {v: k for k, v in self.used_nodes.items()}
        for idx, n in self.used_nodes.items():
            node = self.nodes.get(n)
            parent = node.parent
            if parent:
                idx = name2Id.get(parent)
                node.set_cond(idx)

    def get_codeword(self, node_name=None):

        node = self.nodes.get(node_name, None)
        if not node:
            raise ValueError('{} is not in the tree'.format(node_name))

        codeword = []
        for i in range(len(node.children)):
            chd = node.children[i]
            child = self.nodes.get(chd)
            codeword.append(child.codeword)
        codeword = np.array(codeword)

        return codeword

    def get_nodeId(self, node_name=None):

        node = self.nodes.get(node_name, None)
        if not node:
            raise ValueError('{} is not in the tree'.format(node_name))

        return node.node_id

    def get_parent(self, node_name=None):

        node = self.nodes.get(node_name, None)
        if not node:
            raise ValueError('{} is not in the tree'.format(node_name))

        return node.parent


def write_file(file_name, data_list):

    with open(file_name, 'w') as f:
        for data in data_list:
            f.write('{},{},{}\n'.format(data[1][0], data[1][1], data[0]))


