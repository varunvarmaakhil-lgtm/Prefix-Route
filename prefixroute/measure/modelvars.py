"""Print shell variable assignments from configs/model.json (used by cluster/measure.sbatch)."""
import json

m = json.load(open("configs/model.json"))
print(f"KEY={m['key']} MAXLEN={m['max_model_len']} MNBT={m['max_num_batched_tokens']} "
      f"GPUUTIL={m['gpu_memory_utilization']} QUANT={m['quantization']} PBLOCKS={m['pressure_num_gpu_blocks']}")
