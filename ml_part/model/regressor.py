import torch
import torch.nn as nn
from torch.autograd import Variable

class MLPRegressor(nn.Module):
    """MLP for regression (over multiple tasks) from molecule representations.

    Parameters
    ----------
    in_feats : int
        Number of input molecular graph features
    hidden_feats : int
        Hidden size for molecular graph representations
    n_tasks : int
        Number of tasks, also output size
    dropout : float
        The probability for dropout. Default to 0, i.e. no dropout is performed.
    """
    def __init__(self, in_feats, hidden_feats, n_tasks, dropout=0.):
        super(MLPRegressor, self).__init__()

        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feats, hidden_feats),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_feats),
            nn.Linear(hidden_feats, n_tasks)
        )

    def forward(self, h):
        """Predict for regression.

        Parameters
        ----------
        h : FloatTensor of shape (B, M3)
            * B is the number of molecules in a batch
            * M3 is the input molecule feature size, must match in_feats in initialization

        Returns
        -------
        FloatTensor of shape (B, n_tasks)
        """
        return self.predict(h)

class BaseGNNRegressor(nn.Module):
    """GNN based model for multitask molecular property prediction.
    We assume all tasks are regression problems.

    Parameters
    ----------
    readout_feats : int
        Size for molecular representations
    n_tasks : int
        Number of prediction tasks
    regressor_hidden_feats : int
        Hidden size in MLP regressor
    dropout : float
        The probability for dropout. Default to 0, i.e. no dropout is performed.
    """
    def __init__(self, readout_feats, n_tasks, regressor_hidden_feats=128, dropout=0.):
        super(BaseGNNRegressor, self).__init__()

        self.device = torch.device("cpu")

        self.regressor = MLPRegressor(readout_feats, regressor_hidden_feats, n_tasks, dropout)

    def forward(self, bg, node_feats, edge_feats):
        """Multi-task prediction for a batch of molecules

        Parameters
        ----------
        bg : DGLGraph
            DGLGraph for a batch of B graphs
        node_feats : FloatTensor of shape (N, D0)
            Initial features for all nodes in the batch of graphs
        edge_feats : FloatTensor of shape (M, D1)
            Initial features for all edges in the batch of graphs

        Returns
        -------
        FloatTensor of shape (B, n_tasks)
            Prediction for all tasks on the batch of molecules
        """
        # Update node representations
        feats = self.gnn(bg, node_feats, edge_feats)

        # Compute molecule features from atom features
        h_g = self.readout(bg, feats)

        # Multi-task prediction
        return self.regressor(h_g)
    
    def lrp(self, bg, node_feats, edge_feats):
        node_feats_original = node_feats.clone()
        edge_feats_original = edge_feats.clone()

        all_node_relevances = {}
        all_edge_relevances = {}
        # node relevance
        for node_feat_index in range(len(node_feats)):
            
            self.zero_grad()

            x_node = node_feats_original.clone()
            x_node = Variable(x_node.data, requires_grad=True)

            h0 = x_node

            mask = torch.zeros(x_node.shape).to(self.device)
            mask[node_feat_index] = 1

            x_node = x_node * mask + (1 - mask) * x_node.data
            
            # forward
            feats = self.gnn(bg, x_node, edge_feats_original)
            h_g = self.readout(bg, feats)

            # x_cloned = h_g.repeat(2, 1)
            predictions = self.regressor(h_g)
            logP_prediction = predictions[0][-1]

            # backward
            logP_prediction.backward(retain_graph=True)

            all_node_relevances[node_feat_index] = h0.data * h0.grad
            h0.grad.data.zero_()

        # edge relevance
        for edge_feat_index in range(len(edge_feats)):
            
            self.zero_grad()

            x_edge = edge_feats_original.clone()
            x_edge = Variable(x_edge.data, requires_grad=True)

            e0 = x_edge

            mask = torch.zeros(x_edge.shape).to(self.device)
            mask[edge_feat_index] = 1

            x_edge = x_edge * mask + (1 - mask) * x_edge.data
            
            # forward
            feats = self.gnn(bg, node_feats_original, x_edge)
            h_g = self.readout(bg, feats)

            predictions = self.regressor(h_g)
            logP_prediction = predictions[0][-1]

            # backward
            logP_prediction.backward(retain_graph=True)

            all_edge_relevances[edge_feat_index] = e0.data * e0.grad
            e0.grad.data.zero_()

        # TODO: lrp for smarts
        
        # nodes relevance preprocessing
        for idx, rel in all_node_relevances.items():
            relevance_score = rel.data.sum().item()
            all_node_relevances[idx] = relevance_score

        # edges relevance preprocessing
        for idx, rel in all_edge_relevances.items():
            relevance_score = rel.data.sum().item()
            all_edge_relevances[idx] = relevance_score
        
        return all_node_relevances, all_edge_relevances

class BaseGNNRegressorBypass(nn.Module):
    """This architecture uses one GNN for each task (task-speicifc) and one additional GNN shared
    across all tasks. To predict for each task, we feed the input to both the task-specific GNN
    and the task-shared GNN. The resulted representations of the two GNNs are then concatenated
    and fed to a task-specific forward NN.

    Parameters
    ----------
    readout_feats : int
        Size for molecular representations
    n_tasks : int
        Number of prediction tasks
    regressor_hidden_feats : int
        Hidden size in MLP regressor
    dropout : float
        The probability for dropout. Default to 0, i.e. no dropout is performed.
    """
    def __init__(self, readout_feats, n_tasks, regressor_hidden_feats=128, dropout=0.):
        super(BaseGNNRegressorBypass, self).__init__()

        self.n_tasks = n_tasks
        self.task_gnns = nn.ModuleList()
        self.readouts = nn.ModuleList()
        self.regressors = nn.ModuleList()

        for _ in range(n_tasks):
            self.regressors.append(
                MLPRegressor(readout_feats, regressor_hidden_feats, 1, dropout))

    def forward(self, bg, node_feats, edge_feats):
        """Multi-task prediction for a batch of molecules

        Parameters
        ----------
        bg : DGLGraph
            DGLGraph for a batch of B graphs
        node_feats : FloatTensor of shape (N, D0)
            Initial features for all nodes in the batch of graphs
        edge_feats : FloatTensor of shape (M, D1)
            Initial features for all edges in the batch of graphs

        Returns
        -------
        FloatTensor of shape (B, n_tasks)
            Prediction for all tasks on the batch of molecules
        """
        shared_repr = self.shared_gnn(bg, node_feats, edge_feats)
        predictions = []

        for t in range(self.n_tasks):
            task_repr = self.task_gnns[t](bg, node_feats, edge_feats)
            combined_repr = torch.cat([shared_repr, task_repr], dim=1)
            g_t = self.readouts[t](bg, combined_repr)
            predictions.append(self.regressors[t](g_t))

        # Combined predictions of all tasks
        return torch.cat(predictions, dim=1)
