import os
import csv
import json
import time
import random
import logging
import argparse
import inspect
from datetime import datetime
from pathlib import Path

import gym
import numpy as np
from nle import nethack

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# NLE action mapping - correct compass directions
NLE_ACTIONS = {
    # Basic movements
    "more": 0,        # MiscAction.MORE - for "more" prompts
    "w": 1,       # CompassDirection.N
    "d": 2,        # CompassDirection.E
    "s": 3,       # CompassDirection.S
    "a": 4,        # CompassDirection.W
    "e": 5,   # CompassDirection.NE
    "c": 6,   # CompassDirection.SE
    "z": 7,   # CompassDirection.SW
    "q": 8,   # CompassDirection.NW
    "farnorth": 9,    # CompassDirectionLonger.N
    
    # Extended movements and common actions
    "fareast": 10,    # CompassDirectionLonger.E
    "farsouth": 11,   # CompassDirectionLonger.S
    "farwest": 12,    # CompassDirectionLonger.W
    "farnortheast": 13, # CompassDirectionLonger.NE
    "farsoutheast": 14, # CompassDirectionLonger.SE
    "farsouthwest": 15, # CompassDirectionLonger.SW
    "farnorthwest": 16, # CompassDirectionLonger.NW
    
    # Common actions - using standard indices based on documentation
    "up": 17,       # wait/rest
    "down": 18,  # show inventory
    "wait": 19,     # pick up items
    "kick": 20,     # search
    "eat": 21,       # kick
    "search": 22,        # eat


}

class GameTracker:
    def __init__(self):
        self.steps = 0
        self.total_reward = 0
        self.rewards = []
        self.actions = {}
        self.last_observation = None
    
    def add_step(self, action, reward, observation=None):
        self.steps += 1
        self.total_reward += reward
        self.rewards.append(reward)
        
        if action in self.actions:
            self.actions[action] += 1
        else:
            self.actions[action] = 1
            
        if observation is not None:
            self.last_observation = observation
    
    def get_stats(self):
        return {
            "steps": self.steps,
            "total_reward": self.total_reward,
            "mean_reward": self.total_reward / max(1, self.steps),
            "actions": self.actions
        }


def get_unique_seed(seed=None):
    """Generate a seed if none is provided."""
    if seed is not None:
        return seed
    return int(time.time() * 1000) % 10000


def create_nethack_env(env_name, seed=None):
    """Create a NetHack environment directly without custom wrappers."""
    logging.info(f"Creating environment: {env_name}")
    
    # Configure environment
    kwargs = {
        "observation_keys": [
            "glyphs",
            "blstats",
            "tty_chars",
            "inv_letters",
            "inv_strs",
            "tty_cursor",
            "tty_colors",
        ],
        "savedir": None,  
    }
    
    try:
        env = gym.make(env_name, **kwargs)
        logging.info(f"Environment created: {env}")
        
        logging.info(f"Action space type: {type(env.action_space)}")
        logging.info(f"Action space: {env.action_space}")

        try:
            sample_action = env.action_space.sample()
            logging.info(f"Sampled action: {sample_action}")
            logging.info(f"Sampled action type: {type(sample_action)}")
        except Exception as e:
            logging.error(f"Error sampling action: {e}")
        
        if seed is not None:
            env.seed(seed)
        
        return env
    except Exception as e:
        logging.error(f"Error creating environment: {e}")
        raise


def display_game_state(env, tty_chars):
    """Display the current game state to the player."""
    os.system('cls' if os.name == 'nt' else 'clear')
    
    rows, cols = tty_chars.shape
    ascii_map = ""
    for i in range(rows):
        for j in range(cols):
            ascii_map += chr(tty_chars[i, j])
        ascii_map += "\n"
    
    print("\n=== NETHACK MAP ===")
    print(ascii_map)
    
    print("\n=== CONTROLS ===")
    print("Enter commands directly (e.g., 'north', 'east', 'search', etc.)")
    print("Type 'quit' to end the game")
    print("Type 'help' to see available commands")
    print("Type 'actionlist' to see all available NLE action indices")
    print("Type 'action <num>' to try a specific action number")


def show_help():
    """Display help information with available commands."""
    print("\n=== AVAILABLE COMMANDS ===")
    commands = {
        "north": "move north",
        "east": "move east",
        "south": "move south",
        "west": "move west",
        "northeast": "move northeast",
        "southeast": "move southeast",
        "southwest": "move southwest",
        "northwest": "move northwest",
        "up": "go up a staircase",
        "down": "go down a staircase",
        "wait": "rest one move while doing nothing",
        "search": "search for hidden doors and passages",
        "inventory": "show your inventory",
        "pickup": "pick up things at the current location",
        "kick": "kick an enemy or a locked door or chest",
        "eat": "eat something",
        "open": "open a door",
        "close": "close a door",
        "drop": "drop an item",
        "pray": "pray to the gods",
        "cast": "cast a spell",
        "wield": "wield a weapon",
        "puton": "put on armor/ring",
        "remove": "remove armor/ring",
        "throw": "throw an item",
        "actionlist": "show all NLE action indices",
        "action <num>": "try a specific action by index",
        "help": "show this help message",
        "quit": "quit the game",
    }
    
    for cmd, desc in commands.items():
        print(f"{cmd}: {desc}")
    input("\nPress Enter to continue...")


def show_action_list():
    """Display all the available NLE action indices."""
    print("\n=== NLE ACTION INDICES ===")
    for cmd, idx in sorted(NLE_ACTIONS.items(), key=lambda x: x[1]):
        print(f"{idx}: {cmd}")
    
    print("\nTo try a specific action, type 'action <num>'")
    input("Press Enter to continue...")


def run_human_episode(env_name, task_name, output_dir, max_steps=1000, seed=None):
    """Run a human-controlled episode and record the trajectory."""
    seed = get_unique_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    episode_dir = os.path.join(output_dir, f"human_play_{timestamp}")
    Path(episode_dir).mkdir(exist_ok=True, parents=True)
    
    csv_filename = os.path.join(episode_dir, f"{task_name}_human_play.csv")
    json_filename = os.path.join(episode_dir, f"{task_name}_human_play.json")
    
    file_handler = logging.FileHandler(os.path.join(episode_dir, "play.log"))
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)
    
    episode_log = {
        "task": task_name,
        "seed": seed,
        "human_player": True,
    }
    
    tracker = GameTracker()
    
    try:
        env = create_nethack_env(env_name, seed)
        logging.info(f"Environment created with seed {seed}")
        
    except Exception as e:
        logging.error(f"Failed to create environment: {e}")
        return {"error": str(e), "task": task_name}
    
    try:
        obs = env.reset()
        logging.info("Environment reset successfully")
    except Exception as e:
        logging.error(f"Error during environment reset: {e}")
        return {"error": str(e), "task": task_name}
    
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file, escapechar="˘", quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["Step", "Action", "Action Index", "Observation", "Reward", "Done"])
        
        done = False
        step = 0
        episode_return = 0.0
        
        while not done and step < max_steps:
            if "tty_chars" in obs:
                display_game_state(env, obs["tty_chars"])
            else:
                print("\nCurrent observation:", obs)
            
            print(f"\nStep {step+1}/{max_steps} - Enter command: ", end="")
            action_str = input().strip().lower()
            
            if action_str == "quit":
                print("Quitting game...")
                break
            elif action_str == "help":
                show_help()
                continue
            elif action_str == "actionlist":
                show_action_list()
                continue
            elif action_str.startswith("action "):
                try:
                    action_idx = int(action_str.split()[1])
                    print(f"Using action index: {action_idx}")
                    action = action_idx
                except (ValueError, IndexError) as e:
                    print(f"Error parsing action index: {e}")
                    print("Please use format: action <num>")
                    continue
            else:
                if action_str in NLE_ACTIONS:
                    action = NLE_ACTIONS[action_str]
                    print(f"Using action: {action_str} (index: {action})")
                else:
                    print(f"Unknown command: '{action_str}'. Type 'help' to see available commands.")
                    time.sleep(1)
                    continue
            
            try:
                logging.info(f"Executing action: {action_str} -> {action}")
                next_obs, reward, done_flag, info = env.step(action)
                
                obs_str = str(next_obs)
                if "tty_chars" in next_obs:
                    tty_chars = next_obs["tty_chars"]
                    rows, cols = tty_chars.shape
                    obs_str = ""
                    for i in range(rows):
                        for j in range(cols):
                            obs_str += chr(tty_chars[i, j])
                        obs_str += "\n"
                
                csv_writer.writerow([
                    step,
                    action_str,
                    action,
                    obs_str,
                    reward,
                    done_flag
                ])
                
                tracker.add_step(action_str, reward, next_obs)
                
                episode_return += reward
                obs = next_obs
                done = done_flag
                step += 1
                
                debug_dir = os.path.join(episode_dir, "debug")
                Path(debug_dir).mkdir(exist_ok=True, parents=True)
                with open(os.path.join(debug_dir, f"step_{step:04d}.json"), "w") as f:
                    serializable_obs = {}
                    for k, v in obs.items():
                        if isinstance(v, np.ndarray):
                            serializable_obs[k] = v.tolist()
                        else:
                            serializable_obs[k] = v
                    json.dump(serializable_obs, f, indent=2)
                
                # Log result
                logging.info(f"Step {step}: Action '{action_str}' -> {action}, Reward {reward}, Done {done}")
                
            except Exception as e:
                logging.error(f"Error executing action '{action_str}' -> {action}: {e}")
                print(f"Error executing action: {e}")
                print("Please try a different action.")
                time.sleep(2)  
    
    episode_log.update(tracker.get_stats())
    episode_log["episode_return"] = episode_return
    episode_log["num_steps"] = step
    
    with open(json_filename, "w") as f:
        json.dump(episode_log, f, indent=4)
    
    env.close()
    
    print(f"\nGame over! Episode return: {episode_return}")
    print(f"Trajectory saved to: {episode_dir}")
    
    return episode_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play NetHack with human control and record trajectory")
    parser.add_argument("--env", type=str, default="NetHackScore-v0", 
                        help="NetHack environment to play (default: NetHackScore-v0)")
    parser.add_argument("--task", type=str, default="human_play",
                        help="Task name for recording purposes (default: human_play)")
    parser.add_argument("--output", type=str, default="human_play_results",
                        help="Output directory for trajectories (default: human_play_results)")
    parser.add_argument("--steps", type=int, default=1000,
                        help="Maximum steps per episode (default: 1000)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: None, will generate random seed)")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output).mkdir(exist_ok=True, parents=True)
    
    # Run human episode
    try:
        print(f"Starting NetHack environment: {args.env}")
        print("Loading environment...")
        episode_log = run_human_episode(
            args.env, 
            args.task, 
            args.output,
            max_steps=args.steps,
            seed=args.seed
        )
        print(f"Game completed with score: {episode_log.get('episode_return', 0)}")
    except Exception as e:
        logging.error(f"Error: {e}")
        import traceback
        traceback.print_exc() 