# Data
The data splits can be either accessed with the key "fnames" in `split/{dataset}/{split}.json`, or the pre-generated data lists `data/{dataset}/gt_{split}.txt`. The mapping of class name and class id are in `data/{dataset}/leaf_nodes.npy`.

# Taxonomy
`prepro.py` traces the data hierarchy, and constructs a tree-type taxonomy, which is managed into a `Tree` object. To load the pre-built taxonomy, run
```
>>> import numpy as np
>>> tree = np.load('tree.npy').tolist()
```
To show the taxonomy of the whole tree:
```
>>> tree.show()
root
--aquatic_mammal
----pinniped_mammal
------walrus
------seal
----whale
------baleen_whale
--------humpback
...
```
To show the taxonomy of a subtree (e.g. rooted by "aquatic_mammal"):
```
>>> tree.show('aquatic_mammal')
aquatic_mammal
--pinniped_mammal
----walrus
----seal
--whale
----baleen_whale
------humpback
...
```
To print the information of certain node:
```
>>> print(tree.nodes.get('elephant'))
name=elephant, node_id=71, depth=1, children=
>>> print(tree.nodes.get('bear'))
name=bear, node_id=60, depth=2, children=grizzly,ice_bear
```
These can also be accessed by
```
>>> tree.nodes.get('bear').children
{0: 'grizzly', 1: 'ice_bear'}
>>> tree.nodes.get('grizzly').parent
'bear'
>>> tree.get_parent('grizzly')
'bear'
```
