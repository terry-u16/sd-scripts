accelerate launch `
    --num_processes 1  --num_machines 1 --num_cpu_threads_per_process 2 `
    ./sdxl_train_network.py `
    --config_file `
    ./config/config_lora-20250211-011534.toml 
