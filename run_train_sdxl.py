import shutil
import subprocess

import toml


# Windows 環境では既定エンコーディング (cp932) で開くと UTF-8 の日本語コメントを含む
# TOML ファイル読み込み時に UnicodeDecodeError になる場合があるため、明示的に UTF-8 を指定。
def load_toml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return toml.load(f)


config = load_toml("config/config_template.toml")
config_variables = load_toml("config/config_variables.toml")
config_secret = load_toml("config/config_secret.toml")

config.update(config_variables)
config.update(config_secret)

with open("config/config.toml", "w", encoding="utf-8") as f:
    toml.dump(config, f)

print("config.toml is generated.")

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
