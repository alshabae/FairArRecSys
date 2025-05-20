from typing import List
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, dims: List[int], add_bias=True, act="gelu", apply_layernorm=False, elemwise_affine=False):
        super().__init__()
        self._activation = self._get_activation(act)
        self._apply_layernorm = apply_layernorm
        self._elemwise_affine = elemwise_affine
        self._add_bias = add_bias
        self._model = self._create_model(dims)

    def _create_model(self, dims):
        layers = nn.ModuleList()
        for i in range(1, len(dims)):
            layer = nn.Linear(dims[i-1], dims[i]) if self._add_bias else nn.Linear(dims[i-1], dims[i], bias=False)
            layers.append(layer)

            if i < len(dims) - 1:
                if self._apply_layernorm:
                    layers.append(nn.LayerNorm(dims[i], elementwise_affine=self._elemwise_affine))

                layers.append(self._activation)
        
        return nn.Sequential(*layers)

    def _get_activation(self, act):
        if act == 'gelu':
            return nn.GELU()
        elif act == 'relu':
            return nn.ReLU()
        elif act == 'mish':
            return nn.Mish()
        elif act == 'tanh':
            return nn.Tanh()
        else:
            raise NotImplementedError


    def forward(self, input):
        return self._model(input)

class DotCompressScoringModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], act='gelu'):
        super(DotCompressScoringModel, self).__init__()
        self.dot_compress_weight = nn.Parameter(torch.empty(2, input_dim // 2))
        nn.init.xavier_normal_(self.dot_compress_weight)
        
        self.dot_compress_bias = nn.Parameter(torch.zeros(input_dim // 2))

        self.dims = [input_dim] + hidden_dims + [1]
        self.output_layer = MLP(self.dims, apply_layernorm=True, elemwise_affine=True)
    
    def forward(self, set_embeddings, item_embeddings):
        all_embeddings = torch.stack([set_embeddings, item_embeddings], dim=1)
        combined_representation = torch.matmul(all_embeddings, torch.matmul(all_embeddings.transpose(1, 2), self.dot_compress_weight) + self.dot_compress_bias).flatten(1)
        output = self.output_layer(combined_representation)
        return output


#region BARD ------------------------------------
class BARDNNItemModel(nn.Module):
    def __init__(self,
        num_movie_ids,
        feat_embed_dim=64,
        dense_feat_input_dim=324,
        output_embed_dim=128,
        combine_op='cat'
    ):
        super(BARDNNItemModel, self).__init__()
        self.id_embeddings = nn.Embedding(num_movie_ids, feat_embed_dim)

        self.output_embed_dim = output_embed_dim
        self.combine_op = combine_op
        self.act = nn.GELU()

        self.output_mlp = self._create_output_mlp(1*feat_embed_dim if combine_op == 'cat' else feat_embed_dim, output_embed_dim)
        
    
    def _create_output_mlp(self, first_layer_dim, output_embed_dim):
        return nn.Sequential(nn.Linear(first_layer_dim, 128), 
                             nn.LayerNorm(128, elementwise_affine=False), 
                             self.act, 
                             nn.Linear(128, 64),
                             nn.LayerNorm(64, elementwise_affine=False),
                             self.act, 
                             nn.Linear(64, output_embed_dim))

    def forward(self, movie_ids):
        id_embeddings = self.id_embeddings(movie_ids)

        return self.act(self.output_mlp(id_embeddings))

class BARDNNUserModel(nn.Module):
    def __init__(self,
        num_user_ids,
        feat_embed_dim=64,
        output_embed_dim=128,
        combine_op='cat',
    ):
        super(BARDNNUserModel, self).__init__()
        self.id_embeddings = nn.Embedding(num_user_ids, embedding_dim=feat_embed_dim)

        self.output_embed_dim = output_embed_dim
        self.combine_op = combine_op
        self.act = nn.GELU()

        self.output_mlp = self._create_output_mlp(1*feat_embed_dim if combine_op == 'cat' else feat_embed_dim, output_embed_dim)
    
    def _create_output_mlp(self, first_layer_dim, output_embed_dim):
        return nn.Sequential(nn.Linear(first_layer_dim, 128), 
                             nn.LayerNorm(128, elementwise_affine=False), 
                             self.act, 
                             nn.Linear(128, 64),
                             nn.LayerNorm(64, elementwise_affine=False),
                             self.act, 
                             nn.Linear(64, output_embed_dim))

    
    def forward(self, user_ids):
        id_embeddings = self.id_embeddings(user_ids)
        
        return self.act(self.output_mlp(id_embeddings))
    
class BARDSparseNN(nn.Module):
    def __init__(self,
        num_user_ids,
        num_movie_ids,
        feat_embed_dim=96, # changed from 96 to match what we had before and allow reviews
        dense_feat_embed_dim=384,
        output_embed_dim=128, # for now it should match feat_embed_dim
        combine_op='cat',
    ):
        super(BARDSparseNN, self).__init__()
        self.output_embed_dim = output_embed_dim

        self.user_embedding_model = BARDNNUserModel(num_user_ids, 
                                                               feat_embed_dim=feat_embed_dim, 
                                                               output_embed_dim=output_embed_dim,
                                                               combine_op=combine_op)

        self.item_embedding_model = BARDNNItemModel(num_movie_ids,
                                                               feat_embed_dim=feat_embed_dim,
                                                               dense_feat_input_dim=dense_feat_embed_dim,
                                                               output_embed_dim=output_embed_dim,
                                                               combine_op=combine_op)
        
        self.act = nn.GELU()
        self.scoring_model = DotCompressScoringModel(output_embed_dim, [128, 64])
        


    def forward(self, 
        user_ids = None,
        movie_ids = None,
        user_embeddings_precomputed = None,
        item_embeddings_precomputed = None,
    ):
        user_embeddings = self.user_embedding_model(user_ids) if user_embeddings_precomputed is None \
            else user_embeddings_precomputed
        
        item_embeddings = self.item_embedding_model(movie_ids) if item_embeddings_precomputed is None \
            else item_embeddings_precomputed

        return self.act(self.scoring_model(user_embeddings, item_embeddings))
#endregion