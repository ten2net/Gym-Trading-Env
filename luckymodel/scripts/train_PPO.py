import sys
import copy # Added for deepcopy

from stable_baselines3 import PPO
sys.path.append("../")
import datetime
import argparse
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, ProgressBarCallback
# StopTrainingOnRewardThreshold was removed

# from sb3_contrib import RecurrentPPO # Removed as it's not used
# from stable_baselines3.common.utils import get_schedule_fn # Removed as it's not used
import warnings
from envs.env import make_env
from callbacks.profit_curriculum_callback import ProfitCurriculumCallback,EpisodeMetricsCallback,EntropyScheduler
from scripts.utils import RobustCosineSchedule
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium.wrappers import RecordEpisodeStatistics
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", message="sys.meta_path is None, Python is likely shutting down")

# --- Default Configuration ---
# Centralized configuration for the training script.
# These values can be overridden by command-line arguments.
default_config = {
    # --- Data and Symbols ---
    'symbol_train': '300059',  # Stock symbol for training
    # 'symbol_train': '300520',  # Stock symbol for training
    'symbol_eval': '300308',   # Stock symbol for evaluation

    # --- Learning Rate Schedule ---
    'learning_rate_schedule_params': {
        'initial_lr': 2e-5,    # Starting learning rate
        'final_lr': 2e-4,      # Peak learning rate after warmup
        'warmup_ratio': 0.1,  # Proportion of total steps for linear warmup
    },

    # --- PPO Algorithm Parameters ---
    'ppo_params': {
        'n_steps': 288,          # Number of steps to run for each environment per update, 6天数据(48 bars/day × 6)
        'batch_size': 32,        # Minibatch size for PPO updates
        'n_epochs': 10,           # Number of epochs when optimizing the surrogate loss
        'clip_range_initial': 0.1,# Initial clipping parameter for PPO
        'clip_range_final': 0.05,  # Final clipping parameter (linearly annealed)
        'ent_coef': 0.05,         # Entropy coefficient for exploration
        'gamma': 0.92,            # Discount factor for future rewards
        'device': "cpu",          # Device to use for training ('cpu' or 'cuda')
        'seed': 42,               # Random seed for reproducibility
    },

    # --- Environment Settings ---
    'common_env_params': {
        'window_size': None,          # Observation window size (number of past days)
        'eval': False,             # Base environment mode (set true in create_env for eval)
        'positions': [0, 0.5, 1],  # Allowed positions (e.g., short, neutral, long)
        'trading_fees': 0.01/100,  # Percentage trading fee
        'portfolio_initial_value': 1000000, # Initial portfolio value
        'max_episode_duration': 480,    # Max steps per episode
        'render_mode': "logs",     # Environment render mode
        'verbose': 0,              # Environment verbosity level
    },

    # --- Path Settings ---
    'paths': {
        'tensorboard_log_dir': "../logs/stock_trading2/", # Directory for TensorBoard logs
        'best_model_save_path': "../checkpoints/best_models/", # Directory to save best models during evaluation
        'model_save_prefix': "rppo_trading_model",       # Prefix for final saved model filename
        'tb_log_name': "ppo_300059",                 # Name for the TensorBoard log run
    },

    # --- Training Control ---
    'total_timesteps': int(4e6), # Total number of timesteps to train the agent
    'target_return': 0.02,       # Target return for the training environment's reward function
    'stop_loss': 0.1,           # Stop loss threshold for the training environment
    'eval_stop_loss': 0.1,       # Stop loss threshold for the evaluation environment

    # --- Callback Configurations ---
    # Evaluation Callback (integrates Early Stopping)
    'eval_callback_params': {
        'eval_freq': 48 * 10,        # How_often to perform evaluation (in steps)
        'n_eval_episodes': 10000,    # Number of episodes to run for evaluation
        'verbose': 0,                # Verbosity level for evaluation callback
        'deterministic': True,       # Whether to use deterministic actions for evaluation
        'render': False,             # Whether to render the environment during evaluation
    },
    # Early Stopping (used by EvalCallback)
    'early_stopping_params': {
        'max_no_improvement_evals': 100, # Number of evaluations with no improvement before stopping
        'min_evals': 50,                 # Minimum number of evaluations before early stopping can occur
        'verbose': 1,                    # Verbosity level for early stopping
    },
    # Entropy Scheduler Callback
    'entropy_scheduler_params': {
        'initial_value': 0.05, # Initial entropy coefficient
        'final_value': 0.02,  # Final entropy coefficient (linearly annealed)
    },
    # Parameters for model.learn()
    'model_learn_params':{
        'progress_bar': True,        # Whether to show SB3's internal progress bar
        'log_interval': 10,          # Log training information every N episodes
        'reset_num_timesteps': True, # Whether to reset the timestep counter at the beginning of learning
    },
    # --- Callback Toggles ---
    'use_eval_callback': False,         # Master toggle for EvalCallback (and nested EarlyStopping)
    'use_progress_bar_callback': False, # Master toggle for the separate ProgressBarCallback
}


def create_env(symbol: str, common_env_params: dict, target_return: float, stop_loss: float, is_eval: bool = False, eval_stop_loss: float = 0.1):
    """
    Creates, wraps, and prepares a trading environment for training or evaluation.

    Args:
        symbol (str): The stock symbol (e.g., '300059') for which to create the environment.
        common_env_params (dict): A dictionary of parameters common to both training and evaluation environments.
        target_return (float): The target return percentage used in the reward calculation.
        stop_loss (float): The stop-loss percentage for the training environment.
        is_eval (bool, optional): Flag indicating if the environment is for evaluation. 
                                  If True, `eval_stop_loss` is used. Defaults to False.
        eval_stop_loss (float, optional): The stop-loss percentage specifically for the evaluation environment. 
                                          Defaults to 0.1.

    Returns:
        stable_baselines3.common.vec_env.VecNormalize: The fully wrapped and normalized vectorized environment.
    """
    # Create a mutable copy of common parameters to avoid modifying the original dict
    env_params = common_env_params.copy()
    
    # Set environment-specific parameters based on whether it's for evaluation or training
    env_params['target_return'] = target_return 
    if is_eval:
        env_params['stop_loss'] = eval_stop_loss # Apply specific stop-loss for evaluation
        env_params['eval'] = True                # Set 'eval' flag for make_env if it uses it
    else:
        env_params['stop_loss'] = stop_loss      # Apply training stop-loss
        env_params['eval'] = False               # Ensure 'eval' flag is False for training

    # Create the base environment using the make_env utility
    env = make_env(symbol=symbol, **env_params)
    
    # Wrap the environment with standard wrappers for statistics, vectorization, and normalization
    env = RecordEpisodeStatistics(env)  # Records episode statistics (reward, length)
    env = DummyVecEnv([lambda: env]) 
     # env = SubprocVecEnv([lambda: env for _ in range(8)])# Converts the environment to a vectorized environment
    # env = VecNormalize(env, norm_obs=True, norm_reward=False,clip_reward=10) # Normalizes observations, but not rewards
    
    return env


def train(config: dict):
    """
    Sets up and trains a Proximal Policy Optimization (PPO) agent for stock trading.

    The function configures the training environment, evaluation environment,
    learning rate scheduler, PPO model, and various callbacks based on the provided
    configuration dictionary. It then initiates the training process and returns
    the trained model.

    Args:
        config (dict): A dictionary containing all necessary parameters for setting up
                       and running the training process. This includes parameters for
                       data, PPO algorithm, environment settings, paths, training control,
                       and callbacks. See `default_config` for an example structure.

    Returns:
        stable_baselines3.PPO: The trained PPO model.
    """
    # --- Environment Setup ---
    # Prepare common parameters for environment creation, ensuring window_size is correctly passed
    env_creation_params = config['common_env_params'].copy()

    # Create the training environment
    print(f"Creating training environment for symbol: {config['symbol_train']}")
    train_env = create_env(
        symbol=config['symbol_train'],
        common_env_params=env_creation_params,
        target_return=config['target_return'],
        stop_loss=config['stop_loss'],
        is_eval=False
    )
   
    # Create the evaluation environment
    print(f"Creating evaluation environment for symbol: {config['symbol_eval']}")
    eval_env = create_env(
        symbol=config['symbol_eval'],
        common_env_params=env_creation_params,
        target_return=config['target_return'], # Eval might use the same target_return or a different one if specified
        stop_loss=config['stop_loss'],         # Base stop_loss from training config
        is_eval=True,
        eval_stop_loss=config['eval_stop_loss'] # Specific stop_loss for evaluation
    )
    
    # --- Learning Rate Scheduler Setup ---
    lr_params = config['learning_rate_schedule_params']
    lr_scheduler = RobustCosineSchedule(
        initial_lr=lr_params['initial_lr'],
        final_lr=lr_params['final_lr'],
        warmup_ratio=lr_params['warmup_ratio']
    )   
    
    # --- PPO Model Instantiation ---
    ppo_p = config['ppo_params']
    # Define a linear schedule function for PPO's clip_range
    def linear_schedule(initial_value, final_value):
        """Returns a function that provides a linearly changing value based on progress."""
        def func(progress_remaining):
            """progress_remaining goes from 1.0 to 0.0"""
            return final_value + (initial_value - final_value) * progress_remaining
        return func   

    print(f"Torch CUDA available: {torch.cuda.is_available()}, Using device: {ppo_p['device']}")
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256])  # 策略网络和价值网络均为2层256单元
    )  
    model = PPO(
        "MlpPolicy",  # Standard Multi-Layer Perceptron policy
        train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=lr_scheduler,  
        n_steps=ppo_p['n_steps'],          # Number of steps per environment per update
        batch_size=ppo_p['batch_size'],    # Minibatch size
        n_epochs=ppo_p['n_epochs'],        # Number of optimization epochs per update
        clip_range=linear_schedule(ppo_p['clip_range_initial'], ppo_p['clip_range_final']), # PPO clipping
        ent_coef=ppo_p['ent_coef'],        # Entropy coefficient for encouraging exploration
        gamma=ppo_p['gamma'],              # Discount factor for future rewards
        verbose=0,                         # Verbosity level (0 for no output, 1 for info)
        device=ppo_p['device'],            # PyTorch device
        seed=ppo_p['seed'],                # Random seed
        tensorboard_log=config['paths']['tensorboard_log_dir'] # Directory for TensorBoard logs
    )
    
    # --- Callbacks Setup ---
    print("Setting up callbacks...")
    active_callbacks = []

    # Entropy Scheduler Callback: Dynamically adjusts the entropy coefficient during training.
    entropy_p = config['entropy_scheduler_params']
    # active_callbacks.append(EntropyScheduler(
    #     initial_value=entropy_p['initial_value'], 
    #     final_value=entropy_p['final_value']
    # ))

    # Profit Curriculum Callback: (Assumed to implement a curriculum learning strategy based on profit)
    # 自定义参数（从0.03到0.15，在3e6步内线性增长）
    curriculum_callback = ProfitCurriculumCallback(
        max_steps=config['total_timesteps'],
        initial_target=0.02,
        final_target=0.12,
        ent_coef_initial=entropy_p['initial_value'],
        ent_coef_min=entropy_p['final_value'],
        ent_coef_restore_ratio=0.5
    )    
    active_callbacks.append(curriculum_callback)

    # Episode Metrics Callback: (Assumed to log custom metrics per episode)
    active_callbacks.append(EpisodeMetricsCallback())

    # Evaluation Callback with Early Stopping:
    # Periodically evaluates the model on the evaluation environment.
    # If `callback_on_new_best` is set, it triggers early stopping if performance plateaus.
    if config.get('use_eval_callback', True): 
        early_stop_p = config['early_stopping_params']
        early_stop_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=early_stop_p['max_no_improvement_evals'],
            min_evals=early_stop_p['min_evals'],
            verbose=early_stop_p['verbose']
        )
        eval_cb_p = config['eval_callback_params']
        eval_callback = EvalCallback(
            eval_env=eval_env,
            callback_on_new_best=early_stop_callback, # Integrate early stopping
            best_model_save_path=config['paths']['best_model_save_path'],
            eval_freq=eval_cb_p['eval_freq'],
            n_eval_episodes=eval_cb_p['n_eval_episodes'],
            verbose=eval_cb_p['verbose'],
            deterministic=eval_cb_p['deterministic'],
            render=eval_cb_p['render']
        )
        active_callbacks.append(eval_callback)
        print("EvalCallback with EarlyStopping enabled.")

    # ProgressBar Callback: Displays a progress bar during training.
    if config.get('use_progress_bar_callback', True): 
        active_callbacks.append(ProgressBarCallback())
        print("ProgressBarCallback enabled.")
      
    # --- Model Training ---
    print(f"Starting model training for {config['total_timesteps']} timesteps...")
    learn_p = config['model_learn_params']
    model.learn(
        total_timesteps=config['total_timesteps'],
        progress_bar=learn_p['progress_bar'], # Use SB3's built-in TQDM progress bar
        log_interval=learn_p['log_interval'],   # Log every N episodes
        tb_log_name=config['paths']['tb_log_name'], # Name of the run for TensorBoard
        callback=active_callbacks,              # List of callbacks to use during training
        reset_num_timesteps=learn_p['reset_num_timesteps'] # Whether to reset timestep counter
    )
    print(f"Training completed. Total timesteps: {model.num_timesteps}")
    return model


def parse_args():
    """
    Parses command-line arguments for the training script.

    Utilizes `default_config` to set default values for arguments, ensuring consistency
    and providing informative help messages that reflect these defaults.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Train a PPO model for stock trading.")
    
    # General training arguments
    parser.add_argument('--symbol_train', type=str, default=default_config['symbol_train'],
                        help=f"Stock symbol for training (default: {default_config['symbol_train']})")
    parser.add_argument('--symbol_eval', type=str, default=default_config['symbol_eval'],
                        help=f"Stock symbol for evaluation (default: {default_config['symbol_eval']})")
    parser.add_argument('--window_size', type=int, default=default_config['common_env_params']['window_size'],
                        help=f"Observation window size for the environment (default: {default_config['common_env_params']['window_size']})")
    parser.add_argument('--target_return', type=float, default=default_config['target_return'],
                        help=f"Target return percentage for training rewards (default: {default_config['target_return']})")
    parser.add_argument('--stop_loss', type=float, default=default_config['stop_loss'],
                        help=f"Stop-loss percentage for training environment (default: {default_config['stop_loss']})")
    parser.add_argument('--total_timesteps', type=int, default=default_config['total_timesteps'],
                        help=f"Total number of timesteps for training (default: {default_config['total_timesteps']})")
    
    # Path configuration arguments
    parser.add_argument('--tensorboard_log_dir', type=str, default=default_config['paths']['tensorboard_log_dir'],
                        help=f"Directory for TensorBoard logs (default: \"{default_config['paths']['tensorboard_log_dir']}\")")
    parser.add_argument('--best_model_save_path', type=str, default=default_config['paths']['best_model_save_path'],
                        help=f"Directory to save best models during evaluation (default: \"{default_config['paths']['best_model_save_path']}\")")
    parser.add_argument('--model_save_prefix', type=str, default=default_config['paths']['model_save_prefix'],
                        help=f"Filename prefix for the final saved model (default: \"{default_config['paths']['model_save_prefix']}\")")
    parser.add_argument('--tb_log_name', type=str, default=default_config['paths']['tb_log_name'],
                        help=f"Specific name for the TensorBoard log run (default: \"{default_config['paths']['tb_log_name']}\")")
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # --- Argument Parsing ---
    # Parse command-line arguments. Defaults are sourced from `default_config`.
    cli_args = parse_args()

    # --- Configuration Setup ---
    # Create a deep copy of the default configuration to allow modifications.
    run_config = copy.deepcopy(default_config)

    # Override default configuration with any values provided via command-line arguments.
    run_config['symbol_train'] = cli_args.symbol_train
    run_config['symbol_eval'] = cli_args.symbol_eval
    run_config['total_timesteps'] = cli_args.total_timesteps
    run_config['target_return'] = cli_args.target_return
    run_config['stop_loss'] = cli_args.stop_loss
    
    # Update nested dictionary for common_env_params
    run_config['common_env_params']['window_size'] = cli_args.window_size
    
    # Update nested dictionary for paths
    run_config['paths']['tensorboard_log_dir'] = cli_args.tensorboard_log_dir
    run_config['paths']['best_model_save_path'] = cli_args.best_model_save_path
    run_config['paths']['model_save_prefix'] = cli_args.model_save_prefix
    run_config['paths']['tb_log_name'] = cli_args.tb_log_name
    
    # Example: How to potentially override other config values if more CLI args were added
    # if cli_args.device is not None: # Assuming 'device' could be a CLI argument
    #    run_config['ppo_params']['device'] = cli_args.device

    # --- Model Training ---
    # Train the PPO model using the prepared configuration.
    print("Initializing training process...")
    model = train(config=run_config)
    
    # --- Model Saving ---
    # Save the trained model with a timestamp.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    model_save_path = f"{run_config['paths']['model_save_prefix']}_{timestamp}.zip"
    model.save(model_save_path)
    print(f"Trained model saved to: {model_save_path}")
