uv run benchmarking.py --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12 --warmup_steps 5 --timing_steps 10
uv run benchmarking.py --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12 --warmup_steps 5 --timing_steps 10 --backward
uv run benchmarking.py --d_model 1024 --d_ff 4096 --num_layers 24 --num_heads 16 --warmup_steps 5 --timing_steps 10
uv run benchmarking.py --d_model 1024 --d_ff 4096 --num_layers 24 --num_heads 16 --warmup_steps 5 --timing_steps 10 --backward
uv run benchmarking.py --d_model 1280 --d_ff 5120 --num_layers 36 --num_heads 20 --warmup_steps 5 --timing_steps 10
uv run benchmarking.py --d_model 1280 --d_ff 5120 --num_layers 36 --num_heads 20 --warmup_steps 5 --timing_steps 10 --backward
uv run benchmarking.py --d_model 1600 --d_ff 6400 --num_layers 48 --num_heads 25 --warmup_steps 5 --timing_steps 10
uv run benchmarking.py --d_model 1600 --d_ff 6400 --num_layers 48 --num_heads 25 --warmup_steps 5 --timing_steps 10 --backward
uv run benchmarking.py --d_model 2560 --d_ff 10240 --num_layers 32 --num_heads 32 --warmup_steps 5 --timing_steps 10
uv run benchmarking.py --d_model 2560 --d_ff 10240 --num_layers 32 --num_heads 32 --warmup_steps 5 --timing_steps 10 --backward

uv run nsys profile -o result --force-overwrite true python benchmarking.py --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12 --warmup_steps 5 --timing_steps 10 --context_length 128
