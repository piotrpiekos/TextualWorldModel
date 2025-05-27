# TextualWorldModel

# ManualWorldModel

This is a simpler case in which we want to test manually written world models to verify whether it is even possible to improve performance of LLM with textual world models.

## Usage

For now we focus on Minihack, Custom world models are specific to the **task** (like `MiniHack-Quest-Easy-v0`). To choose a task, modify `BALROG/balrog/config/config.yaml` file, l86, list all the tasks (probably just one) you want to run.

Then in the `BALROG/balrog/world_models/current.json` create a dictionary entrance with a world model for a given task.

Next, run `world model eval` configuration from the `.vscode` included configrations. This will run **only** minihack with a given task with augmented textual world model. Also the configuration already includes the OPENAI_API_TOKEN, but don't spread it