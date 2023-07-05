import jax
import numpy as np
from itertools import count
from environment import ContTermEnv
from dqn.dqn import DQN
from dqn.policy import EpsilonGreedy
from dqn.model import create_fwd_fn, Network

from absl import app
from absl import flags



FLAGS = flags.FLAGS


flags.DEFINE_integer('max_steps', 10000, 'maximum steps to make before done', lower_bound=1)
flags.DEFINE_integer('batch_size', 256, 'batch_size', lower_bound=1)
flags.DEFINE_integer('length', 25, 'length of the terminal', lower_bound=1)
flags.DEFINE_integer('breadth', 25, 'breadth of the terminal', lower_bound=1)
flags.DEFINE_integer('height', 5, 'max height of the raw of containers', lower_bound=1)
flags.DEFINE_integer('num_destinations', 50, 'number of destination points', lower_bound=1)
flags.DEFINE_integer('num_containers', 300, 'number containers to load/discharge', lower_bound=1)
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




def launch(argv):
    env = ContTermEnv(max_steps=FLAGS.max_steps,
                      batch_size=FLAGS.batch_size,
                      length=FLAGS.length,
                      breadth=FLAGS.breadth,
                      height=FLAGS.height,
                      num_destinations=FLAGS.num_destinations,
                      num_containers=FLAGS.num_containers)

    n_states = env.observation_space
    n_act = env.action_space
    rng = jax.random.PRNGKey(46)
    policy = EpsilonGreedy(epsilon=lambda x: FLAGS.epsilon_min + (FLAGS.epsilon - FLAGS.epsilon_min) * np.exp(-FLAGS.epsilon_decay/ FLAGS.num_containers * x))
    model = create_fwd_fn(Network, num_actions=FLAGS.length*FLAGS.breadth)

    agent = DQN(rng, n_states.shape, n_act.n, FLAGS.gamma, 10000, policy, model, FLAGS.lr)

    ep_rewards, losses = agent.train_on_env(env, 10000000, FLAGS.batch_size, FLAGS.target_freq, FLAGS.report)


if __name__ == '__main__':
    local_devices = jax.local_devices()
    global_devices = jax.devices()
    print(local_devices)
    print(global_devices)
    
    with jax.default_device(global_devices[0]):
        app.run(launch)
