import torch
import torch.nn as nn
from gym import spaces
import numpy as np
from itertools import count
from environment import ContTermEnv
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from absl import app
from absl import flags



FLAGS = flags.FLAGS


flags.DEFINE_integer('max_steps', 10000, 'maximum steps to make before done', lower_bound=1)
flags.DEFINE_integer('batch_size', 128, 'batch_size', lower_bound=1)
flags.DEFINE_integer('length', 32, 'length of the terminal', lower_bound=1)
flags.DEFINE_integer('breadth', 32, 'breadth of the terminal', lower_bound=1)
flags.DEFINE_integer('height', 7, 'max height of the raw of containers', lower_bound=1)
flags.DEFINE_integer('num_destinations', 50, 'number of destination points', lower_bound=1)
flags.DEFINE_integer('num_containers', 1000, 'number containers to load/discharge', lower_bound=1)
flags.DEFINE_integer('target_freq', 2, 'target_freq', lower_bound=1)
flags.DEFINE_integer('report', 1, 'printing info frequency by epochs', lower_bound=1)
flags.DEFINE_float('wd', 1e-4, 'weight_decay')
flags.DEFINE_float('lr', 3e-4, 'learning_rate')
flags.DEFINE_float('gamma', 0.99, 'gamma')
flags.DEFINE_float('epsilon', 1.0, 'epsilon')
flags.DEFINE_float('epsilon_min', 0.01, 'epsilon_min')
flags.DEFINE_float('epsilon_decay', 0.01, 'epsilon_decay')
flags.DEFINE_string('name', 'Masked_VAE', 'Model name')
flags.DEFINE_string('params_dir', 'weights', 'directory where models parameters are saved')
flags.DEFINE_string('dataset', 'ROSE', 'Name of the dataset')
flags.DEFINE_bool('var', True, 'calculate variance of the dataset')
flags.DEFINE_bool('pretrained', True, 'load pretrained parameters to the model')


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        out = self.linear(self.cnn(observations))
        return out


class Custom1D(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv1d(n_input_channels, 48, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(48, 96, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        out = self.linear(self.cnn(observations))
        return out
    

class MyMSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MyMSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"

        d_head = int(d / n_heads)
        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        # Sequences has shape (N, seq_length, token_dim)
        # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
        # And come back to    (N, seq_length, item_dim)  (through concatenation)
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])
    

class MyViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(MyViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MyMSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out


class CustomTransform(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(self, 
                 observation_space: spaces.Box, 
                 n_patches: int, 
                 hidden_d: int, 
                 n_heads: int, 
                 blocks: int,
                 mlp_ratio: int, 
                 features_dim: int,
                 device: str
                 ):
        super().__init__(observation_space, features_dim)
        # Attributes
        self.chw = observation_space.shape # (C, H, W)
        self.n_patches = n_patches
        self.n_blocks = blocks
        self.hidden_d = hidden_d
        self.n_heads = n_heads
        self.mlp_ratio = mlp_ratio
        self.device = device

        assert self.chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        assert self.chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        self.patch_size = (self.chw[1] / n_patches, self.chw[2] / n_patches)

        # 1) Linear mapper
        self.input_d = int(self.chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        # 3) Positional embedding
        '''To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or 
        sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor)
        '''
        self.pos_embed = nn.Parameter(self.get_positional_embeddings(self.n_patches ** 2, self.hidden_d).clone().detach())
        self.pos_embed.requires_grad = False

        self.blocks = nn.ModuleList([MyViTBlock(hidden_d, n_heads, self.mlp_ratio) for _ in range(self.n_blocks)])

        self.linear = nn.Sequential(nn.Linear(self.n_patches ** 2 * self.hidden_d, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        patches = self.patchify(observations, self.n_patches).to(self.device)
        tokens = self.linear_mapper(patches)

        # Adding positional embedding
        pos_embed = self.pos_embed.repeat(observations.shape[0], 1, 1)
        out = tokens + pos_embed

        for block in self.blocks:
            out = block(out)

        out = out.reshape(out.shape[0], -1)
        out = self.linear(out)

        return out
    
    def patchify(self, images, n_patches):
        n, c, h, w = images.shape

        assert h == w, "Patchify method is implemented for square images only"

        patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
        patch_size = h // n_patches

        for idx, image in enumerate(images):
            for i in range(n_patches):
                for j in range(n_patches):
                    patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                    patches[idx, i * n_patches + j] = patch.flatten()
        return patches

    def get_positional_embeddings(self, sequence_length, d):
        result = torch.ones(sequence_length, d)
        for i in range(sequence_length):
            for j in range(d):
                result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
        return result


def launch(argv):
    new_logger = configure('./logs', ["stdout", "tensorboard"])


    env = make_vec_env("ContTermEnv-v2", n_envs=32, vec_env_cls=SubprocVecEnv)
    policy_cnn = dict(
                        features_extractor_class=CustomCNN,
                        features_extractor_kwargs=dict(features_dim=1024)
                        )
    policy_1d = dict(
                        features_extractor_class=Custom1D,
                        features_extractor_kwargs=dict(features_dim=1024)
                        )
    policy_trnsf = dict(
                        features_extractor_class=CustomTransform,
                        features_extractor_kwargs=dict(n_patches=8, 
                                                       hidden_d=32, 
                                                       n_heads=4, 
                                                       blocks=6, 
                                                       mlp_ratio=4, 
                                                       features_dim=1024,
                                                       device='cuda')
                        )
    model = A2C(
                "CnnPolicy",
                # 'MlpPolicy',
                env, 
                policy_kwargs=policy_trnsf,
                n_steps=5,
                learning_rate=1e-4, 
                # gae_lambda=0.95, 
                ent_coef=0.0005, 
                device="cuda",
                max_grad_norm=0.5
                )
    model.set_logger(new_logger)
    model.learn(total_timesteps=1e13, log_interval=200)


if __name__ == '__main__':
    app.run(launch)
        
