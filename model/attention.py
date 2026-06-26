import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.Key = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.Query = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.Value = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        
        # shapes
        batch_size, context_length, embedding_dim = embedded.shape

        # linear transformations on the embeddings (matmuls to get Q,K,V)
        K = self.Key(embedded)
        Q = self.Query(embedded)
        V = self.Value(embedded)

        # obtain the attention dimension 
        attention_dim = K.shape[2]

        # compute scores
        K_trans = K.permute(0, 2, 1)
        scores = torch.matmul(Q, K_trans)/(attention_dim**0.5)


        # apply masking to prevent past tokens influencing the upcoming ones
        mask = torch.tril(torch.ones(context_length, context_length)).bool()

        # now apply the mask in-place
        masked_scores = scores.masked_fill(~mask, float('-inf'))

        # now apply over the column direction 
        softmax = torch.softmax(masked_scores, dim=2)

        return torch.round(torch.matmul(softmax, V), decimals=4)

