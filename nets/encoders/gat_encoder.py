
import numpy as np
import torch
from torch._C import ErrorReport
import torch.nn.functional as F
from torch import nn
import math

import dgl
import dgl.function as fn

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math


import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math



import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math

'''
class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input, mask=None):
        return input + self.module(input, mask=mask)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, input_dim, embed_dim=None, val_dim=None, key_dim=None):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """
        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
                     Attention mask should contain 1 if attention is not possible (additive attention)
        """
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Optionally apply mask to prevent attention
        if mask is not None:
            compatibility[mask[None, :, :, :].expand_as(compatibility)] = -1e10

        attn = F.softmax(compatibility, dim=-1)

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        return out


class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch', learn_norm=True, track_norm=False):
        super(Normalization, self).__init__()

        self.normalizer = {
            "layer": nn.LayerNorm(embed_dim, elementwise_affine=learn_norm),
            "batch": nn.BatchNorm1d(embed_dim, affine=learn_norm, track_running_stats=track_norm)
        }.get(normalization, None)

    def forward(self, input, mask=None):
        if self.normalizer:
            return self.normalizer(
                input.view(-1, input.size(-1))
            ).view(*input.size())
        else:
            return input


class PositionWiseFeedforward(nn.Module):

    def __init__(self, embed_dim, feed_forward_dim):
        super(PositionWiseFeedforward, self).__init__()
        self.sub_layers = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_dim, bias=True),
            nn.ReLU(),
            nn.Linear(feed_forward_dim, embed_dim, bias=True),
        )

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input, mask=None):
        return self.sub_layers(input)


class MultiHeadAttentionLayer(nn.Module):
    """Implements a configurable Transformer layer
    References:
        - W. Kool, H. van Hoof, and M. Welling. Attention, learn to solve routing problems! In International Conference on Learning Representations, 2019.
        - M. Deudon, P. Cournut, A. Lacoste, Y. Adulyasak, and L.-M. Rousseau. Learning heuristics for the tsp by policy gradient. In International Conference on the Integration of Constraint Programming, Artificial Intelligence, and Operations Research, pages 170–181. Springer, 2018.
    """

    def __init__(self, n_heads, embed_dim, feed_forward_dim, 
                 norm='batch', learn_norm=True, track_norm=False):
        super(MultiHeadAttentionLayer, self).__init__()

        self.self_attention = SkipConnection(
            MultiHeadAttention(
                    n_heads=n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
            )
        self.norm1 = Normalization(embed_dim, norm, learn_norm, track_norm)

        self.positionwise_ff = SkipConnection(
            PositionWiseFeedforward(
                   embed_dim=embed_dim,
                   feed_forward_dim=feed_forward_dim
                )
            )
        self.norm2 = Normalization(embed_dim, norm, learn_norm, track_norm)

    def forward(self, h, mask):
        h = self.self_attention(h, mask=mask)
        h = self.norm1(h, mask=mask)
        h = self.positionwise_ff(h, mask=mask)
        h = self.norm2(h, mask=mask)
        return h


class GraphAttentionEncoder(nn.Module):

    def __init__(self, n_layers, n_heads, hidden_dim, norm='batch', 
                 learn_norm=True, track_norm=False, *args, **kwargs):
        super(GraphAttentionEncoder, self).__init__()
        
        feed_forward_hidden = hidden_dim * 4
        
        self.layers = nn.ModuleList([
            MultiHeadAttentionLayer(n_heads, hidden_dim, feed_forward_hidden, norm, learn_norm, track_norm)
                for _ in range(n_layers)
        ])

    def forward(self, x, graph):
        for layer in self.layers:
            x = layer(x, graph)
        return x




'''



'''

class BatchNormNode(nn.Module):
    """Batch normalization for node features.
    """

    def __init__(self, hidden_dim):
        super(BatchNormNode, self).__init__()
        self.batch_norm = nn.BatchNorm1d(hidden_dim, track_running_stats=False)

    def forward(self, x):
        """
        Args:
            x: Node features (batch_size, hidden_dim, num_nodes)
        Returns:
            x_bn: Node features after batch normalization (batch_size, hidden_dim, num_nodes)
        """
        x_bn = self.batch_norm(x) # B x H x N
        return x_bn


class BatchNormEdge(nn.Module):
    """Batch normalization for edge features.
    """

    def __init__(self, hidden_dim):
        super(BatchNormEdge, self).__init__()
        self.batch_norm = nn.BatchNorm2d(hidden_dim, track_running_stats=False)

    def forward(self, e):
        """
        Args:
            e: Edge features (batch_size, hidden_dim, num_nodes, num_nodes)
        Returns:
            e_bn: Edge features after batch normalization (batch_size, hidden_dim, num_nodes, num_nodes)
        """
        e_bn = self.batch_norm(e) # B x H x N x N
        return e_bn




class NodeFeatures(nn.Module):
    """Convnet features for nodes.
    
    Using `sum` aggregation:
        x_i = U*x_i +  sum_j [ gate_ij * (V*x_j) ]
    
    Using `mean` aggregation:
        x_i = U*x_i + ( sum_j [ gate_ij * (V*x_j) ] / sum_j [ gate_ij] )
    """
    
    def __init__(self, hidden_dim, aggregation="mean"):
        super(NodeFeatures, self).__init__()
        self.aggregation = aggregation
        self.U = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_gate):
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)
            edge_gate: Edge gate values (batch_size, num_nodes, num_nodes, hidden_dim)
        Returns:
            x_new: Convolved node features (batch_size, num_nodes, hidden_dim)
        """
        Ux = self.U(x)  # B x H x V
        Vx = self.V(x)  # B x H x V
        Vx = Vx.unsqueeze(2)  # extend Vx from "B x H x V" to "B x H x 1 x V"
        gateVx = edge_gate * Vx  # B x H x V x V
        if self.aggregation=="mean":
            x_new = Ux + torch.sum(gateVx, dim=3) / (1e-20 + torch.sum(edge_gate, dim=3))  # B x H x V
        elif self.aggregation=="sum":
            x_new = Ux + torch.sum(gateVx, dim=3)  # B x H x V
        return x_new


class EdgeFeatures(nn.Module):
    """Convnet features for edges.
    e_ij = U*e_ij + V*(x_i + x_j)
    """

    def __init__(self, hidden_dim):
        super(EdgeFeatures, self).__init__()
        self.U = nn.Linear(hidden_dim, hidden_dim,hidden_dim)
        self.V = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, e):
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)
            e: Edge features (batch_size, num_nodes, num_nodes, hidden_dim)
        Returns:
            e_new: Convolved edge features (batch_size, num_nodes, num_nodes, hidden_dim)
        """
        Ue = self.U(e) # B x H x V x V
        Vx = self.V(x) # B x H x V
        Wx = Vx.unsqueeze(2)  # Extend Vx from "B x H x V" to "B x H x 1 x V"
        Vx = Vx.unsqueeze(3)  # extend Vx from "B x H x V" to "B x H x V x 1"
        e_new = Ue + Vx + Wx
        return 



        
'''



class MLPLayer(nn.Module):
    """Simple MLP layer with ReLU activation
    """

    def __init__(self, hidden_dim, norm="layer", learn_norm=True, track_norm=False):
        """
        Args:
            hidden_dim: Hidden dimension size (int)
            norm: Feature normalization scheme ("layer"/"batch"/None)
            learn_norm: Whether the normalizer has learnable affine parameters (True/False)
            track_norm: Whether batch statistics are used to compute normalization mean/std (True/False)
        """
        super(MLPLayer, self).__init__()

        self.hidden_dim = hidden_dim
        self.norm = norm
        self.learn_norm = learn_norm
        self.track_norm=track_norm


        self.U1 = nn.Linear(hidden_dim,hidden_dim,bias=True)

        self.maxpooling1 = nn.MaxPool1d(kernel_size=1, stride=1)

        self.V1 = nn.Linear(hidden_dim,hidden_dim,bias=True)

        self.norm1 = {
            "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
            "batch": nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
        }.get(self.norm, None)

        

    def forward(self, x):
        batch_size, num_nodes, hidden_dim = x.shape
        x_in = x
        layer_outputs1=[]
        

        # Linear transformation
        x = self.U1(x)

        x=self.maxpooling1(x)

        x = self.V1(x)

        # Normalize features
        x = self.norm1(
            x.view(batch_size*num_nodes, hidden_dim)
        ).view(batch_size, num_nodes, hidden_dim) if self.norm else x



        # Apply non-linearity
        x = F.relu(x)

        # Make residual connection
        x = x_in + x
        return x










class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input, mask=None):
        return input + self.module(input, mask=mask)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, input_dim, embed_dim=None, val_dim=None, key_dim=None):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))


        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.init_parameters()


        self.MLPlayers1 = nn.ModuleList(
            MLPLayer(embed_dim, norm='layer', learn_norm=True, track_norm=False) for _ in range(3)
        )

        #self.eps1 = nn.Parameter(torch.zeros(embed_dim))
        self.eps1 = nn.Parameter(torch.zeros(embed_dim, embed_dim))





    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """
        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
                     Attention mask should contain 1 if attention is not possible (additive attention)
        """
        if h is None:
            h = q  # compute self-attention
            #x_in=q

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"



        for layer in self.MLPlayers1:
            h = layer(torch.matmul(h.view(batch_size* graph_size, input_dim), self.eps1).view(batch_size,  graph_size, input_dim))
            


        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        
        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)
        


        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)

        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)

        K = torch.matmul(hflat, self.W_key).view(shp)
        

        V = torch.matmul(hflat, self.W_val).view(shp)
        


        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)

        #compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        #compatibility = self.norm_factor *  Q
        compatibility = torch.matmul(self.norm_factor * Q , self.norm_factor * K.transpose(2, 3) )
        

        # Optionally apply mask to prevent attention
        if mask is not None:
            compatibility[mask[None, :, :, :].expand_as(compatibility)] = -1e10


        
        # Edge convolution
        #e_tmp = self.edge_feat(h, compatibility)  # B x H x V x V
        # Compute edge gates
        #compatibility = F.sigmoid(e_tmp)
        


        attn = F.softmax(compatibility, dim=-1) #compatibility = egde *conv layer 


        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        #LSTM
        #out, _ = self.rnn(out)

        #out= self.output_layer(out)

        #out=F.softmax(out,dim=1)

        return out


class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch', learn_norm=True, track_norm=False):
        super(Normalization, self).__init__()

        self.normalizer = {
            "layer": nn.LayerNorm(embed_dim, elementwise_affine=learn_norm),
            "batch": nn.BatchNorm1d(embed_dim, affine=learn_norm, track_running_stats=track_norm)
        }.get(normalization, None)

    def forward(self, input, mask=None):
        if self.normalizer:
            return self.normalizer(
                input.view(-1, input.size(-1))
            ).view(*input.size())
        else:
            return input


class PositionWiseFeedforward(nn.Module):

    def __init__(self, embed_dim, feed_forward_dim):
        super(PositionWiseFeedforward, self).__init__()
        self.sub_layers = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_dim, bias=True),
            nn.ReLU(),
            nn.Linear(feed_forward_dim, embed_dim, bias=True),
        )

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input, mask=None):
        return self.sub_layers(input)


class MultiHeadAttentionLayer(nn.Module):
    """Implements a configurable Transformer layer
    References:
        - W. Kool, H. van Hoof, and M. Welling. Attention, learn to solve routing problems! In International Conference on Learning Representations, 2019.
        - M. Deudon, P. Cournut, A. Lacoste, Y. Adulyasak, and L.-M. Rousseau. Learning heuristics for the tsp by policy gradient. In International Conference on the Integration of Constraint Programming, Artificial Intelligence, and Operations Research, pages 170–181. Springer, 2018.
    """

    def __init__(self, n_heads, embed_dim, feed_forward_dim, 
                 norm='batch', learn_norm=True, track_norm=False):
        super(MultiHeadAttentionLayer, self).__init__()

        self.self_attention = SkipConnection(
            MultiHeadAttention(
                    n_heads=n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
            )
        self.norm1 = Normalization(embed_dim, norm, learn_norm, track_norm)

        self.positionwise_ff = SkipConnection(
            PositionWiseFeedforward(
                   embed_dim=embed_dim,
                   feed_forward_dim=feed_forward_dim
                )
            )
        self.norm2 = Normalization(embed_dim, norm, learn_norm, track_norm)




    def forward(self, h, mask):

        h = self.self_attention(h, mask=mask)
        h = self.norm1(h, mask=mask)
        h = self.positionwise_ff(h, mask=mask)
        h = self.norm2(h, mask=mask)
            
        return h


class GraphAttentionEncoder(nn.Module):

    def __init__(self,n_layers, n_heads, hidden_dim, norm='batch', 
                 learn_norm=True, track_norm=False, *args, **kwargs):
        super(GraphAttentionEncoder, self).__init__()
        
        feed_forward_hidden = hidden_dim * 4
        
        self.layers = nn.ModuleList([
            MultiHeadAttentionLayer(n_heads, hidden_dim, feed_forward_hidden, norm, learn_norm, track_norm)
                for _ in range(n_layers)
        ])

        self.n_layers=n_layers

        """An implementation of Jumping Knowledge Network (arxiv 1806.03536) which
          combine layers with 
                            concatenation.
        """
        #self.last_linear = torch.nn.Linear(hidden_dim*n_layers,hidden_dim)


        """An implementation of Jumping Knowledge Network (arxiv 1806.03536) which
            combine layers with 
                            Maxpool.
        """

        self.last_linear = torch.nn.Linear(hidden_dim,hidden_dim)

        self.hidden_dim = hidden_dim
        self.norm = norm
        self.learn_norm = learn_norm
        
        self.norm1 = {
            "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
            "batch": nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
        }.get(self.norm, None)


        #self.rnn = torch.nn.LSTM(input_size = self.hidden_dim, 
        #                         hidden_size = self.hidden_dim,
        #                         num_layers = 1, 
        #                         batch_first = True,)
        #                         #bidirectional=True)



        #self.output_layer = nn.Linear(hidden_dim, hidden_dim)

        #nn.init.xavier_normal_(self.rnn.weight_ih_l0)
        #nn.init.orthogonal_(self.rnn.weight_hh_l0)


    def forward(self, x, graph):
        batch_size, graph_size, input_dim = x.size()
        layer_outputs = []
        n_layers=self.n_layers

        for layer in self.layers:
          x=layer(x,graph)
          layer_outputs.append(x)



        #Max
        x = torch.stack(layer_outputs,dim=1)
        x = torch.max(x, dim=1)[0]
        
        #Concat
        #x = torch.cat(layer_outputs, dim=2)

        
        x_in=x

        #Linner Tranformer
        
        x=self.last_linear(x)
        
        
        # Normalize features
        x = self.norm1(
            x.view(batch_size*graph_size, input_dim)
        ).view(batch_size, graph_size, input_dim)if self.norm else x


        # Apply non-linearity
        x = F.relu(x)

        # Make residual connection
        x = x_in + x
        
        


        #LSTM
        #x, self.hidden_dim = self.rnn(x)


        #x= self.output_layer(x)

        #x=F.softmax(x,dim=1)

        #SE layer 
        #b, a, c = x.size()
        #y = self.avg_pool(x).view(c)
        #y = self.fc(y).view(c)
        #return x * y.expand_as(x)
        
        return x




