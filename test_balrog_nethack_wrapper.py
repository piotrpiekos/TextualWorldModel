import gym
import minihack

from balrog.environments.nle import NLELanguageWrapper
from balrog.environments.wrappers import GymV21CompatibilityV0, NLETimeLimit

env = NLELanguageWrapper(gym.make("MiniHack-Corridor-R5-v0",
        observation_keys=[
            "glyphs",
            "blstats",
            "tty_chars",
            "inv_letters",
            "inv_strs",
            "tty_cursor",
            "tty_colors",
        ],
), vlm=False)

env = GymV21CompatibilityV0(env=NLETimeLimit(env), render_mode=None)

obsv, info = env.reset(seed=42)
#action = env.action_space.sample()  

obsv, reward, done, info = env.step("north")
print(obsv)

