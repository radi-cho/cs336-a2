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

uv run nsys profile -o c128_small --force-overwrite true python benchmarking.py --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12 --warmup_steps 5 --timing_steps 10 --context_length 128
uv run nsys profile -o c128_small_b --force-overwrite true python benchmarking.py --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12 --warmup_steps 5 --timing_steps 10 --backward --context_length 128
uv run nsys profile -o c128_medium --force-overwrite true python benchmarking.py --d_model 1024 --d_ff 4096 --num_layers 24 --num_heads 16 --warmup_steps 5 --timing_steps 10 --context_length 128
uv run nsys profile -o c128_medium_b --force-overwrite true python benchmarking.py --d_model 1024 --d_ff 4096 --num_layers 24 --num_heads 16 --warmup_steps 5 --timing_steps 10 --backward --context_length 128
uv run nsys profile -o c128_large --force-overwrite true python benchmarking.py --d_model 1280 --d_ff 5120 --num_layers 36 --num_heads 20 --warmup_steps 5 --timing_steps 10 --context_length 128
uv run nsys profile -o c128_large_b --force-overwrite true python benchmarking.py --d_model 1280 --d_ff 5120 --num_layers 36 --num_heads 20 --warmup_steps 5 --timing_steps 10 --backward --context_length 128
uv run nsys profile -o c128_xl --force-overwrite true python benchmarking.py --d_model 1600 --d_ff 6400 --num_layers 48 --num_heads 25 --warmup_steps 5 --timing_steps 10 --context_length 128
uv run nsys profile -o c128_xl_b --force-overwrite true python benchmarking.py --d_model 1600 --d_ff 6400 --num_layers 48 --num_heads 25 --warmup_steps 5 --timing_steps 10 --backward --context_length 128
uv run nsys profile -o c128_2b --force-overwrite true python benchmarking.py --d_model 2560 --d_ff 10240 --num_layers 32 --num_heads 32 --warmup_steps 5 --timing_steps 10 --context_length 128
uv run nsys profile -o c128_2b_b --force-overwrite true python benchmarking.py --d_model 2560 --d_ff 10240 --num_layers 32 --num_heads 32 --warmup_steps 5 --timing_steps 10 --backward --context_length 128

uv run nsys profile -o c256_small --force-overwrite true python benchmarking.py --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12 --warmup_steps 5 --timing_steps 10 --context_length 256
uv run nsys profile -o c256_small_b --force-overwrite true python benchmarking.py --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12 --warmup_steps 5 --timing_steps 10 --backward --context_length 256
uv run nsys profile -o c256_medium --force-overwrite true python benchmarking.py --d_model 1024 --d_ff 4096 --num_layers 24 --num_heads 16 --warmup_steps 5 --timing_steps 10 --context_length 256
uv run nsys profile -o c256_medium_b --force-overwrite true python benchmarking.py --d_model 1024 --d_ff 4096 --num_layers 24 --num_heads 16 --warmup_steps 5 --timing_steps 10 --backward --context_length 256
uv run nsys profile -o c256_large --force-overwrite true python benchmarking.py --d_model 1280 --d_ff 5120 --num_layers 36 --num_heads 20 --warmup_steps 5 --timing_steps 10 --context_length 256
uv run nsys profile -o c256_large_b --force-overwrite true python benchmarking.py --d_model 1280 --d_ff 5120 --num_layers 36 --num_heads 20 --warmup_steps 5 --timing_steps 10 --backward --context_length 256
uv run nsys profile -o c256_xl --force-overwrite true python benchmarking.py --d_model 1600 --d_ff 6400 --num_layers 48 --num_heads 25 --warmup_steps 5 --timing_steps 10 --context_length 256
uv run nsys profile -o c256_xl_b --force-overwrite true python benchmarking.py --d_model 1600 --d_ff 6400 --num_layers 48 --num_heads 25 --warmup_steps 5 --timing_steps 10 --backward --context_length 256
uv run nsys profile -o c256_2b --force-overwrite true python benchmarking.py --d_model 2560 --d_ff 10240 --num_layers 32 --num_heads 32 --warmup_steps 5 --timing_steps 10 --context_length 256
uv run nsys profile -o c256_2b_b --force-overwrite true python benchmarking.py --d_model 2560 --d_ff 10240 --num_layers 32 --num_heads 32 --warmup_steps 5 --timing_steps 10 --backward --context_length 256

uv run nsys profile -o c512_small --force-overwrite true python benchmarking.py --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12 --warmup_steps 5 --timing_steps 10 --context_length 512
uv run nsys profile -o c512_small_b --force-overwrite true python benchmarking.py --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12 --warmup_steps 5 --timing_steps 10 --backward --context_length 512
uv run nsys profile -o c512_medium --force-overwrite true python benchmarking.py --d_model 1024 --d_ff 4096 --num_layers 24 --num_heads 16 --warmup_steps 5 --timing_steps 10 --context_length 512
uv run nsys profile -o c512_medium_b --force-overwrite true python benchmarking.py --d_model 1024 --d_ff 4096 --num_layers 24 --num_heads 16 --warmup_steps 5 --timing_steps 10 --backward --context_length 512
uv run nsys profile -o c512_large --force-overwrite true python benchmarking.py --d_model 1280 --d_ff 5120 --num_layers 36 --num_heads 20 --warmup_steps 5 --timing_steps 10 --context_length 512
uv run nsys profile -o c512_large_b --force-overwrite true python benchmarking.py --d_model 1280 --d_ff 5120 --num_layers 36 --num_heads 20 --warmup_steps 5 --timing_steps 10 --backward --context_length 512
uv run nsys profile -o c512_xl --force-overwrite true python benchmarking.py --d_model 1600 --d_ff 6400 --num_layers 48 --num_heads 25 --warmup_steps 5 --timing_steps 10 --context_length 512
uv run nsys profile -o c512_xl_b --force-overwrite true python benchmarking.py --d_model 1600 --d_ff 6400 --num_layers 48 --num_heads 25 --warmup_steps 5 --timing_steps 10 --backward --context_length 512
uv run nsys profile -o c512_2b --force-overwrite true python benchmarking.py --d_model 2560 --d_ff 10240 --num_layers 32 --num_heads 32 --warmup_steps 5 --timing_steps 10 --context_length 512
uv run nsys profile -o c512_2b_b --force-overwrite true python benchmarking.py --d_model 2560 --d_ff 10240 --num_layers 32 --num_heads 32 --warmup_steps 5 --timing_steps 10 --backward --context_length 512

uv run nsys profile -o c1024_small --force-overwrite true python benchmarking.py --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12 --warmup_steps 5 --timing_steps 10 --context_length 1024
uv run nsys profile -o c1024_small_b --force-overwrite true python benchmarking.py --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12 --warmup_steps 5 --timing_steps 10 --backward --context_length 1024
uv run nsys profile -o c1024_medium --force-overwrite true python benchmarking.py --d_model 1024 --d_ff 4096 --num_layers 24 --num_heads 16 --warmup_steps 5 --timing_steps 10 --context_length 1024
uv run nsys profile -o c1024_medium_b --force-overwrite true python benchmarking.py --d_model 1024 --d_ff 4096 --num_layers 24 --num_heads 16 --warmup_steps 5 --timing_steps 10 --backward --context_length 1024
uv run nsys profile -o c1024_large --force-overwrite true python benchmarking.py --d_model 1280 --d_ff 5120 --num_layers 36 --num_heads 20 --warmup_steps 5 --timing_steps 10 --context_length 1024
uv run nsys profile -o c1024_large_b --force-overwrite true python benchmarking.py --d_model 1280 --d_ff 5120 --num_layers 36 --num_heads 20 --warmup_steps 5 --timing_steps 10 --backward --context_length 1024
uv run nsys profile -o c1024_xl --force-overwrite true python benchmarking.py --d_model 1600 --d_ff 6400 --num_layers 48 --num_heads 25 --warmup_steps 5 --timing_steps 10 --context_length 1024
uv run nsys profile -o c1024_xl_b --force-overwrite true python benchmarking.py --d_model 1600 --d_ff 6400 --num_layers 48 --num_heads 25 --warmup_steps 5 --timing_steps 10 --backward --context_length 1024
uv run nsys profile -o c1024_2b --force-overwrite true python benchmarking.py --d_model 2560 --d_ff 10240 --num_layers 32 --num_heads 32 --warmup_steps 5 --timing_steps 10 --context_length 1024
uv run nsys profile -o c1024_2b_b --force-overwrite true python benchmarking.py --d_model 2560 --d_ff 10240 --num_layers 32 --num_heads 32 --warmup_steps 5 --timing_steps 10 --backward --context_length 1024

uv run nsys profile -o optim_128_small --force-overwrite true python benchmarking.py --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12 --warmup_steps 5 --timing_steps 10 --context_length 128 --backward --adam_step

uv run benchmarking.py --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12 --warmup_steps 5 --timing_steps 10 --mixed_precision
uv run benchmarking.py --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12 --warmup_steps 5 --timing_steps 10 --backward --mixed_precision
uv run benchmarking.py --d_model 1024 --d_ff 4096 --num_layers 24 --num_heads 16 --warmup_steps 5 --timing_steps 10 --mixed_precision
uv run benchmarking.py --d_model 1024 --d_ff 4096 --num_layers 24 --num_heads 16 --warmup_steps 5 --timing_steps 10 --backward --mixed_precision
uv run benchmarking.py --d_model 1280 --d_ff 5120 --num_layers 36 --num_heads 20 --warmup_steps 5 --timing_steps 10 --mixed_precision
uv run benchmarking.py --d_model 1280 --d_ff 5120 --num_layers 36 --num_heads 20 --warmup_steps 5 --timing_steps 10 --backward --mixed_precision
uv run benchmarking.py --d_model 1600 --d_ff 6400 --num_layers 48 --num_heads 25 --warmup_steps 5 --timing_steps 10 --mixed_precision
uv run benchmarking.py --d_model 1600 --d_ff 6400 --num_layers 48 --num_heads 25 --warmup_steps 5 --timing_steps 10 --backward --mixed_precision
uv run benchmarking.py --d_model 2560 --d_ff 10240 --num_layers 32 --num_heads 32 --warmup_steps 5 --timing_steps 10 --mixed_precision
uv run benchmarking.py --d_model 2560 --d_ff 10240 --num_layers 32 --num_heads 32 --warmup_steps 5 --timing_steps 10 --backward --mixed_precision
