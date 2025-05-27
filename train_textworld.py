import os
import gym
import textworld.gym
from openai import OpenAI

# 1. Register and create the TextWorld environment
game_file = "tw_games/custom_game.z8"
env_id = textworld.gym.register_game(game_file, max_episode_steps=50)
env = textworld.gym.make(env_id)

# 2. Initialize OpenAI client
openai_api_key = 'dupa'
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = "gpt-4"


def build_prompt(observation, feedback, valid_actions):
    """
    Constructs a system+user prompt to guide GPT-4.
    You can enrich this with few-shot examples or game-specific context.
    """
    return [
        {"role": "system", "content": "You are an expert text-game player. Choose the best next action."},
        {"role": "user",
         "content":
            f"Observation:\n{observation}\n\n"
            f"Available commands:\n{valid_actions}\n\n"
            f"Previous feedback:\n{feedback}\n\n"
            "Which action do you take? Please output exactly one command."}
    ]

# 3.1. Start the episode
obs, infos = env.reset()
feedback = ""
done = False

while not done:
    # Render or print the current game state
    env.render()

    # Get list of admissible commands for guidance
    valid_actions = infos.get("admissible_commands", [])

    # Build and send prompt to GPT-4
    messages = build_prompt(obs, feedback, valid_actions)
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,    # low temperature for deterministic play
        max_tokens=50
    )
    action = response.choices[0].message.content.strip()

    # Apply the action
    obs, score, done, infos = env.step(action)
    feedback = infos.get("last_feedback", "")

    print(f"> {action}")     # log chosen action
    print(f"Feedback: {feedback}\nScore: {score}\n")

# 3.2. Episode end
print("Episode finished.")
env.close()