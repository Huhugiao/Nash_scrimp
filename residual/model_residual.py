import importlib
import torch
from mlp import alg_parameters_mlp as base_params
from mlp.model_mlp import Model
from residual.nets_residual import ResidualPolicyNetwork
from residual.alg_parameters_residual import ResidualTrainingParameters


class ResidualModel(Model):
    '''
    Adapter for residual PPO training.
    - 使用 ResidualTrainingParameters 覆盖基础超参
    - 使用 ResidualPolicyNetwork (更小的 MLP + 残差缩放)
    - 在 PPO loss 中加入残差 L2 惩罚 (依靠 base Model 的 hook)
    '''

    def __init__(self, device, global_model=False):
        # 覆盖基础 TrainingParameters，以便父类使用残差超参
        self._sync_training_parameters()
        super().__init__(device, global_model)

        # 替换网络为残差网络
        self.network = ResidualPolicyNetwork().to(device)

        # 残差惩罚系数 (供父类 train 使用)
        self.residual_penalty_coef = ResidualTrainingParameters.RESIDUAL_PENALTY_COEF

        if global_model:
            self.net_optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=ResidualTrainingParameters.lr,
                eps=1e-5,
            )
        self.current_lr = ResidualTrainingParameters.lr
        self.network.train()

    @staticmethod
    def _sync_training_parameters():
        '''将 mlp.alg_parameters_mlp.TrainingParameters 同步为残差配置。'''
        names = [
            'lr', 'LR_FINAL', 'LR_SCHEDULE', 'N_ENVS', 'N_STEPS', 'N_MAX_STEPS',
            'LOG_EPOCH_STEPS', 'MINIBATCH_SIZE', 'N_EPOCHS_INITIAL', 'N_EPOCHS_FINAL',
            'CLIP_RANGE', 'VALUE_CLIP_RANGE', 'ENTROPY_COEF', 'EX_VALUE_COEF',
            'MAX_GRAD_NORM', 'GAMMA', 'LAM', 'OPPONENT_TYPE', 'RANDOM_OPPONENT_WEIGHTS'
        ]
        for name in names:
            if hasattr(ResidualTrainingParameters, name):
                setattr(base_params.TrainingParameters, name, getattr(ResidualTrainingParameters, name))

    def update_learning_rate(self, new_lr):
        super().update_learning_rate(new_lr)
        self.current_lr = new_lr
