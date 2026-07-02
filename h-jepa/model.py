import torch.nn as nn

class HJEPAModel(nn.Module):
    def __init__(self, context_encoder, target_encoder, predictor_network):
        super(HJEPAModel, self).__init__()
        self.context_encoder = context_encoder
        self.target_encoder = target_encoder
        self.predictor_network = predictor_network

    def forward(self, context_view, target_view):
        # Encode the context view to get the context embedding
        context_embedding = self.context_encoder(context_view)

        # Predict the target embedding from the context embedding
        predicted_target_embedding = self.predictor_network(context_embedding)

        # Encode the target view to get the actual target embedding
        actual_target_embedding = self.target_encoder(target_view)

        return predicted_target_embedding, actual_target_embedding