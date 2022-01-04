
import torch

from graphwar.functional.scatter import scatter_add


def conv(edges, x, edge_weight):
    row, col = edges
    src = x[row] * edge_weight.view(-1, 1)
    x = scatter_add(src, col, dim=-2, dim_size=x.size(0))
    return x


def backdoor_edges(g):
    N = g.num_nodes()
    device = g.device
    influence_nodes = torch.arange(N, device=device)

    N_all = N + influence_nodes.size(0)
    trigger_nodes = torch.arange(N, N_all, device=device)

    # 1. edge index of original graph (without selfloops)
    edge_index = torch.stack(g.remove_self_loop().edges(), dim=0)

    # 2. edge index of original graph (with selfloops)
    edge_index_with_self_loop = torch.stack(g.add_self_loop().edges(), dim=0)

    # 3. edge index of trigger nodes conneted to victim nodes with selfloops (with self-loop)
    trigger_edge_index = torch.stack([trigger_nodes, influence_nodes], dim=0)
    diag_index = torch.arange(N_all, device=device).repeat(2, 1)
    trigger_edge_index = torch.cat([trigger_edge_index, trigger_edge_index[[1, 0]], diag_index], dim=1)

    # 4. all edge index with trigger nodes
    augmented_edge_index = torch.cat([edge_index, trigger_edge_index], dim=1)

    d = g.in_degrees().float()
    d_augmented = d.clone()
    d_augmented[influence_nodes] += 1.
    d_augmented = torch.cat([d_augmented, torch.full(trigger_nodes.size(), 2, device=device)])

    d_pow = d.pow(-0.5)
    d_augmented_pow = d_augmented.pow(-0.5)

    edge_weight = d_pow[edge_index_with_self_loop[0]] * d_pow[edge_index_with_self_loop[1]]
    edge_weight_with_trigger = d_augmented_pow[edge_index[0]] * d_pow[edge_index[1]]
    trigger_edge_weight = d_augmented_pow[trigger_edge_index[0]] * d_augmented_pow[trigger_edge_index[1]]
    augmented_edge_weight = torch.cat([edge_weight_with_trigger, trigger_edge_weight], dim=0)

    return (edge_index, edge_weight_with_trigger,
            edge_index_with_self_loop, edge_weight,
            trigger_edge_index, trigger_edge_weight,
            augmented_edge_index, augmented_edge_weight)
