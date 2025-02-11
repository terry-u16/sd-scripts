import subprocess
import shutil

import toml

with open("config/config_template.toml") as f:
    config = toml.load(f)

with open("config/config_variables.toml") as f:
    config_variables = toml.load(f)

with open("config/config_secret.toml") as f:
    config_secret = toml.load(f)


config_variables["sample_every_n_steps"] = config_variables["save_every_n_steps"]
config_variables["network_alpha"] = max(config_variables["network_dim"] / 4, 1)

config.update(config_variables)
config.update(config_secret)

with open("config/config.toml", "w") as f:
    toml.dump(config, f)

print("config.toml is generated.")

shutil.copy("config/prompt.txt", config["sample_prompts"])

args = [
    "accelerate",
    "launch",
    "--num_processes",
    "1",
    "--num_machines",
    "1",
    "--num_cpu_threads_per_process",
    "2",
    "./sdxl_train_network.py",
    "--config_file",
    "./config/config.toml",
]

subprocess.run(args).check_returncode()
