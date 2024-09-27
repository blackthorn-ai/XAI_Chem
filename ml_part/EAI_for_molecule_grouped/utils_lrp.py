import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import torch

def check_if_node_in_subgroup(molecule_subgroups, node):
    for molecule_subgroup in molecule_subgroups:
        if node in molecule_subgroup:
            return True
    return False

def check_if_edge_in_subgroup(molecule_subgroups, start_edge, end_edge):
    for molecule_subgroup in molecule_subgroups:
        if start_edge in molecule_subgroup and end_edge in molecule_subgroup:
            return True
    return False

def get_color(value, cmap_type='coolwarm', vmin=-1, vmax=1):
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_type)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    rgba_color = sm.to_rgba(value)
    return rgba_color

def prepare_node_feats(node_feats, pka1_atom_list, pka2_atom_list, args):
    pka1_atom_list=np.array(pka1_atom_list)
    pka1_atom_list[np.isinf(pka1_atom_list)]=15
    pka2_atom_list=np.array(pka2_atom_list)
    pka2_atom_list[np.isinf(pka2_atom_list)]=0

    pka1_feature = torch.Tensor(pka1_atom_list/11).to(args['device'])
    pka2_feature = torch.Tensor(pka2_atom_list/11).to(args['device'])

    pka1_feature=pka1_feature.unsqueeze(-1)
    pka2_feature=pka2_feature.unsqueeze(-1)

    node_feats = torch.cat([node_feats,pka1_feature,pka2_feature],dim = 1)
    return node_feats