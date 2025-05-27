import gym
import minihack
from nle_language_wrapper import NLELanguageWrapper

from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 2. Call the Chat Completion API
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "Hello!"}
    ]
)
# 3. Extract and print the assistant's reply
assistant_message = response.choices[0].message.content
print(assistant_message)


env = NLELanguageWrapper(gym.make("MiniHack-River-v0",
        observation_keys=[
            "glyphs",
            "blstats",
            "tty_chars",
            "inv_letters",
            "inv_strs",
            "tty_cursor",
            "tty_colors",
        ],
))
obsv = env.reset()
#action = env.action_space.sample()  
obsv, reward, done, info = env.step("north")
print(obsv)