import os
import subprocess
import sys

import toml


CONFIG_TEMPLATE = "config/anima_config_template.toml"
CONFIG_VARIABLES = "config/anima_config_variables.toml"
CONFIG_SECRET = "config/anima_config_secret.toml"
CONFIG_OUTPUT = "config/anima_config.toml"

REQUIRED_PATH_KEYS = [
    "pretrained_model_name_or_path",
    "qwen3",
    "vae",
    "train_data_dir",
]


# Windows 環境では既定エンコーディング (cp932) で開くと UTF-8 の日本語コメントを含む
# TOML ファイル読み込み時に UnicodeDecodeError になる場合があるため、明示的に UTF-8 を指定。
def load_toml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return toml.load(f)


def dump_toml(path: str, config: dict):
    with open(path, "w", encoding="utf-8") as f:
        toml.dump(config, f)


def is_placeholder_path(value: str) -> bool:
    return "\\path\\to\\" in value or "/path/to/" in value


def validate_config(config: dict):
    errors = []

    for key in REQUIRED_PATH_KEYS:
        value = config.get(key)
        if not value:
            errors.append(f"{key} is not set.")
            continue
        if isinstance(value, str) and is_placeholder_path(value):
            errors.append(f"{key} still points to a placeholder: {value}")
            continue
        if key in ("pretrained_model_name_or_path", "qwen3", "vae", "train_data_dir") and not os.path.exists(value):
            errors.append(f"{key} does not exist: {value}")

    for key in ("sample_prompts", "llm_adapter_path", "t5_tokenizer_path"):
        value = config.get(key)
        if value and not os.path.exists(value):
            errors.append(f"{key} does not exist: {value}")

    if errors:
        message = "\n".join(f"- {error}" for error in errors)
        raise RuntimeError(
            "Anima training config is not ready. Edit config/anima_config_secret.toml first:\n" + message
        )


def build_config():
    config = load_toml(CONFIG_TEMPLATE)
    config_variables = load_toml(CONFIG_VARIABLES)
    config_secret = load_toml(CONFIG_SECRET)

    config.update(config_variables)
    config.update(config_secret)
    dump_toml(CONFIG_OUTPUT, config)
    return config


config = build_config()
print(f"{CONFIG_OUTPUT} is generated.")
validate_config(config)

args = [
    sys.executable,
    "-m",
    "accelerate.commands.launch",
    "--num_processes",
    "1",
    "--num_machines",
    "1",
    "--num_cpu_threads_per_process",
    "2",
    "./anima_train_network.py",
    "--config_file",
    f"./{CONFIG_OUTPUT}",
]

env = os.environ.copy()
env.setdefault("PYTHONUTF8", "1")

subprocess.run(args, env=env).check_returncode()
