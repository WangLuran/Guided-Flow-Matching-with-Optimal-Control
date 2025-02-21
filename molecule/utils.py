import pickle

import torch

from equifm.cnf_models import Cnflows, EGNN_dynamics_QM9, DistributionNodes, UniformDequantizer, \
    assert_correctly_masked, assert_mean_zero_with_mask
from qm9.property_prediction import main_qm9_prop
from qm9.property_prediction.prop_utils import get_adj_matrix
from consts import qm9_with_h


def get_flow_model(device):
    with open('equifm/args.pickle', 'rb') as f:
        args = pickle.load(f)
    args.dataset = 'qm9_second_half'
    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = 1
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = 'sum'

    histogram = qm9_with_h['n_nodes']
    in_node_nf = len(qm9_with_h['atom_decoder']) + int(args.include_charges)
    nodes_dist = DistributionNodes(histogram)

    # prop_dist = None
    # if len(args.conditioning) > 0:
    #     prop_dist = DistributionProperty(dataloader_train, args.conditioning)

    if args.condition_time:
        dynamics_in_node_nf = in_node_nf + 1
    else:
        print('Warning: dynamics model is _not_ conditioned on time.')
        dynamics_in_node_nf = in_node_nf

    net_dynamics = EGNN_dynamics_QM9(
        in_node_nf=dynamics_in_node_nf,
        context_node_nf=args.context_node_nf,
        n_dims=3,
        device=device,
        hidden_nf=args.nf,
        act_fn=torch.nn.SiLU(),
        n_layers=args.n_layers,
        attention=args.attention,
        tanh=args.tanh,
        mode=args.model,
        norm_constant=args.norm_constant,
        inv_sublayers=args.inv_sublayers,
        sin_embedding=args.sin_embedding,
        normalization_factor=args.normalization_factor,
        aggregation_method=args.aggregation_method,
    )
    dequantizer = UniformDequantizer()
    cnf = Cnflows(
        dynamics=net_dynamics,
        in_node_nf=in_node_nf,
        n_dims=3,
        timesteps=args.diffusion_steps,
        noise_schedule=args.diffusion_noise_schedule,
        noise_precision=args.diffusion_noise_precision,
        loss_type=args.diffusion_loss_type,
        norm_values=args.normalize_factors,
        include_charges=args.include_charges,
        discrete_path=args.discrete_path,
        cat_loss=args.cat_loss,
        cat_loss_step=args.cat_loss_step,
        on_hold_batch=args.on_hold_batch,
        sampling_method=args.sampling_method,
        weighted_methods=args.weighted_methods,
        ode_method=args.ode_method,
        without_cat_loss=args.without_cat_loss,
        angle_penalty=args.angle_penalty,
    )
    cnf = cnf.to(device)
    dequantizer = dequantizer.to(device)
    cnf.load_state_dict(
        torch.load('equifm/generative_model_ema_0.npy', map_location=device)
    )
    return cnf, nodes_dist, dequantizer, args


def get_classifier(prop, device):
    with open(f'qm9/property_prediction/outputs/exp_class_{prop}/args.pickle', 'rb') as f:
        args_classifier = pickle.load(f)
    args_classifier.device = device
    args_classifier.model_name = 'egnn'
    classifier = main_qm9_prop.get_model(args_classifier)
    classifier_state_dict = torch.load(
        f'qm9/property_prediction/outputs/exp_class_{prop}/best_checkpoint.npy', map_location=device
    )
    classifier.load_state_dict(classifier_state_dict)
    return classifier, args_classifier


def setup_generation(batch_size, nodes_dist, max_n_nodes, device):
    nodesxsample = nodes_dist.sample(batch_size)
    assert int(torch.max(nodesxsample)) <= max_n_nodes
    node_mask = torch.zeros(batch_size, max_n_nodes)
    for i in range(batch_size):
        node_mask[i, 0: nodesxsample[i]] = 1
    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    edge_mask = edge_mask.view(batch_size * max_n_nodes * max_n_nodes, 1).to(device)
    node_mask = node_mask.unsqueeze(2).to(device)
    return node_mask, edge_mask, nodesxsample


def get_classifier_score(z, flow, classifier, deq, node_mask, edge_mask, device):
    # prepare data
    x, h = flow.sample_p_xh_given_z0(deq, z, node_mask)
    one_hot = h["categorical"]
    # assert_correctly_masked(x, node_mask)
    # assert_mean_zero_with_mask(x, node_mask)
    # assert_correctly_masked(one_hot.float(), node_mask)

    # run classifier
    batch_size, n_nodes, _ = x.size()
    atom_positions = x.view(batch_size * n_nodes, -1)
    node_mask = node_mask.view(batch_size * n_nodes, -1)
    nodes = one_hot.view(batch_size * n_nodes, -1)
    edges = get_adj_matrix(n_nodes, batch_size, device)
    pred = classifier(
        h0=nodes, x=atom_positions, edges=edges, edge_attr=None,
        node_mask=node_mask, edge_mask=edge_mask, n_nodes=n_nodes
    )
    return pred, x, one_hot
