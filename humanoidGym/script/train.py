from humanoidGym.envs import *
from humanoidGym.utils import get_args, task_registry

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    args.num_envs = 2500
    args.task="lite_amp"
    args.experiment_name="action_lag"
    train(args)