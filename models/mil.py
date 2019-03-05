import torch
from torch import nn
import numpy as np

from .factory import register_model
from .utils import Flatten

@register_model("mil_classifier")
class MultipleInstanceLearningClassifier(nn.Module):

    def __init__(self, hidden_dims=1024, aggregation_type="mean",
                 att_hidden_dims=128):
        super(MultipleInstanceLearningClassifier, self).__init__()
        self.kernel_size = 5
        self.pool_size = 2
        self.hidden_dims = hidden_dims
        self.att_hidden_dims = att_hidden_dims
        self.aggregation_type = aggregation_type

        # Use padding 2 to keep the input output dimensions the same.
        self.encoder = nn.Sequential(                        # In:  (b, 3, 128, 128)
            nn.Conv2d(3, 8, self.kernel_size, padding=2),    # Out: (b, 8, 128, 128)
            nn.MaxPool2d(self.pool_size),                    # Out: (b, 8, 64, 64)
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, self.kernel_size, padding=2),   # Out: (b, 16, 64, 64)
            nn.MaxPool2d(self.pool_size),                    # Out: (b, 16, 32, 32)
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, self.kernel_size, padding=2),  # Out: (b, 32, 32, 32)
            nn.MaxPool2d(self.pool_size),                    # Out: (b, 32, 16, 16)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, self.kernel_size, padding=2),  # Out: (b, 64, 16, 16)
            nn.MaxPool2d(self.pool_size),                    # Out: (b, 64, 8, 8)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, self.kernel_size, padding=2), # Out: (b, 128, 8, 8)
            nn.MaxPool2d(self.pool_size),                    # Out: (b, 128, 4, 4)
            nn.ReLU(inplace=True),
            Flatten(),
            nn.Linear(128*4*4, self.hidden_dims)             # Out: (b, self.hidden_dims)
        )

        if self.aggregation_type == "attention":
            self.attention = nn.Sequential(
                nn.Linear(self.hidden_dims, self.att_hidden_dims),
                nn.Tanh(),
                nn.Linear(self.att_hidden_dims, 1)
            )

        self.aggregator = self.get_aggregator()

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dims, 1),
            nn.Sigmoid()
        )
    
    def get_aggregator(self):
        if self.aggregation_type == "mean":
            def mean(x):
                return torch.mean(x, dim=0)
            return mean
        elif self.aggregation_type == "attention":
            def att_aggr(x):
                att = self.attention(x)
                att = torch.transpose(att, 1, 0)
                att = nn.functional.softmax(att, dim=1)
                return torch.mm(att, x)
            return att_aggr
        else:
            raise ValueError("Unkown aggregation type {}.".format(self.aggregation_type))
    
    def forward(self, data, case_ids):
        """
            data: torch.Tensor of shape (batch_size, 3, 128, 128)
        """
        embeddings = self.encoder(data) # Out: (batch_size, self.hidden_dims)
        embeddings, cases = self.aggregate(embeddings, case_ids) # Out: (num_cases, self.hidden_dims)
        y_prob = self.classifier(embeddings) # Out: (num_cases,)
        return y_prob, cases
    
    def aggregate(self, embeddings, case_ids):
        """Aggregate embeddings so that we end up with one embeddings for each cases.
        Args:
            embeddings: torch.Tensor of shape (batch_size, self.hidden_dims)
            case_ids: List of length batch_size
        Out:
            torch.Tensor of shape (num_cases, self.hidden_dims)
        """
        case_ids = np.array(case_ids)
        cases = np.unique(case_ids)
        # Make sure new embeddings are on the right device for input to classifier.
        device = self.classifier[0].weight.device
        aggr_embeddings = torch.zeros(len(cases), self.hidden_dims).to(device)
        for i, case in enumerate(cases):
            # Indexes for the given case.
            idx = torch.LongTensor(np.argwhere(case_ids == case).flatten()).to(device)
            aggr_embeddings[i] = self.aggregator(embeddings.index_select(0, idx))
        
        return aggr_embeddings, cases