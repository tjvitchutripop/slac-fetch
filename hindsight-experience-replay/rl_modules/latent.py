import torch
import torch.nn as nn
import torch.nn.functional as F


class encoder(nn.Module):
    def __init__(self, env_params):
        super(encoder, self).__init__()
        # Input Tensor of [B, S, D] to [B, L]
        hidden_dim = 128 
        self.net = nn.Sequential(
            nn.Linear(env_params['obs'] + env_params['goal'], hidden_dim), 
            # nn.GRU(hidden_dim, hidden_dim), # Gated Recurrent Unit
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear( hidden_dim, env_params['latent']),
        )

    def forward(self, input_state):
        latent_state = self.net(input_state)
        return latent_state


class decoder(nn.Module):
    def __init__(self, env_params):
        super(decoder, self).__init__()
        hidden_dim = 32
        self.net = nn.Sequential(
            nn.Linear(env_params['latent'], hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.GRU(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, env_params['obs'])
        )

    def forward(self, latent_state):
        predicted_state = self.net(latent_state)
        return predicted_state
    
# class LatentModel(torch.jit.ScriptModule):
#     def __init__(
#          self,
#         state_shape,
#         action_shape,
#         feature_dim=256,
#         z1_dim=32,
#         z2_dim=256,
#         hidden_units=(256, 256),
#     ):
    