
from mlp.model_mlp import Model
from mlp.nets_residual import ResidualPolicyNetwork
from mlp.alg_parameters_mlp import ResidualRLParameters
import torch

class ResidualModel(Model):
    """
    Adapter model for training ResidualPolicyNetwork using PPO.
    Inherits training logic from Model but swaps the network.
    """
    def __init__(self, device, global_model=False):
        # Initialize parent (will create ProtectingNetMLP temporarily)
        super().__init__(device, global_model)
        
        # Swap network
        self.network = ResidualPolicyNetwork().to(device)
        
        # If global, re-initialize optimizer with new network parameters and Residual LR
        if global_model:
            self.net_optimizer = torch.optim.Adam(
                self.network.parameters(), 
                lr=getattr(ResidualRLParameters, 'RESIDUAL_LR', 3e-4), 
                eps=1e-5
            )
