import pickle
import numpy as np
import pandas as pd
from core.action_space import ACTION_SPACES

# Đường dẫn đến ckpt.pkl (cập nhật đúng đường dẫn tới thư mục log2/0)
ckpt_path = "results/blacknoirmax/BOiLS/results/BO/fpga-4_seq-10_ref-resyn2_act-extended/BOiLS_std_init-10_obj-lut_acq-ei_ard_kernel-ssk_yosys/log2/0/ckpt.pkl"

with open(ckpt_path, "rb") as f:
    ckpt = pickle.load(f)

action_space = ACTION_SPACES['extended']  # Nếu dùng action_space_id=extended

def decode_sequence(seq_indices):
    return ' | '.join([action_space[i].act_id for i in seq_indices])

rows = []
for i, seq in enumerate(ckpt.samples):
    ratio_lut = np.mean(ckpt.full_objs_1[i])
    ratio_level = np.mean(ckpt.full_objs_2[i])
    row = {
        "index": i + 1,
        "seq_id": decode_sequence(seq),
        "ratio_lut": ratio_lut,
        "ratio_level": ratio_level,
        "both": ratio_lut + ratio_level
    }
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("all_sequences.csv", index=False)

print("Top 5 sequences with lowest LUT ratio:")
print(df.sort_values("ratio_lut").head(10))
