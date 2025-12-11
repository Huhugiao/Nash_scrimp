import torch
import numpy as np
import os
import sys

# Ensure we can import from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from targetmaker.target_model import TargetPPO
from targetmaker.target_alg_parameters import TargetNetParameters

class RLTargetPolicy:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.agent = TargetPPO(self.device)
        
        print(f"Loading RL Target Model from: {model_path}")
        if not os.path.exists(model_path):
            raise ValueError(f"Model path not found: {model_path}")
            
        weights = torch.load(model_path, map_location=self.device)
        self.agent.set_weights(weights)
        
    def get_action(self, obs, privileged_state=None):
        # obs is expected to be the flat target observation (72 dim)
        ac = self.agent.evaluate(obs)
        # Clip to ensure valid range [-1, 1] as expected by env wrapper logic usually
        return np.clip(ac, -1.0, 1.0)
