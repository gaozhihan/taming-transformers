import sys
import os
from omegaconf import OmegaConf
import yaml
from PIL import Image
import numpy as np
import time


if not os.path.exists("logs/2020-11-09T13-31-51_sflckr/checkpoints"):
    os.makedirs("logs/2020-11-09T13-31-51_sflckr/checkpoints")
    os.system("wget 'https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fcheckpoints%2Flast.ckpt&dl=1' -O 'logs/2020-11-09T13-31-51_sflckr/checkpoints/last.ckpt'")
if not os.path.exists("logs/2020-11-09T13-31-51_sflckr/configs"):
    os.makedirs("logs/2020-11-09T13-31-51_sflckr/configs")
    os.system("wget 'https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fconfigs%2F2020-11-09T13-31-51-project.yaml&dl=1' -O 'logs/2020-11-09T13-31-51_sflckr/configs/2020-11-09T13-31-51-project.yaml'")
sys.path.append(".")

config_path = "logs/2020-11-09T13-31-51_sflckr/configs/2020-11-09T13-31-51-project.yaml"
config = OmegaConf.load(config_path)
print(yaml.dump(OmegaConf.to_container(config)))

from taming.models.cond_transformer import Net2NetTransformer
model = Net2NetTransformer(**config.model.params)

import torch
ckpt_path = "logs/2020-11-09T13-31-51_sflckr/checkpoints/last.ckpt"
sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
missing, unexpected = model.load_state_dict(sd, strict=False)
model.cuda().eval()
torch.set_grad_enabled(False)

segmentation_path = "data/sflckr_segmentations/norway/25735082181_999927fe5a_b.png"
segmentation = Image.open(segmentation_path)
segmentation = np.array(segmentation)
segmentation = np.eye(182)[segmentation]
segmentation = torch.tensor(segmentation.transpose(2,0,1)[None]).to(dtype=torch.float32, device=model.device)

def show_segmentation(s, save_path="tmp.png"):
  s = s.detach().cpu().numpy().transpose(0,2,3,1)[0,:,:,None,:]
  colorize = np.random.RandomState(1).randn(1,1,s.shape[-1],3)
  colorize = colorize / colorize.sum(axis=2, keepdims=True)
  s = s@colorize
  s = s[...,0,:]
  s = ((s+1.0)*127.5).clip(0,255).astype(np.uint8)
  s = Image.fromarray(s)
  # display(s)
  s.save(save_path)

show_segmentation(segmentation, "loaded_segmentation.png")

c_code, c_indices = model.encode_to_c(segmentation)
print("c_code", c_code.shape, c_code.dtype)
print("c_indices", c_indices.shape, c_indices.dtype)
assert c_code.shape[2]*c_code.shape[3] == c_indices.shape[0]
segmentation_rec = model.cond_stage_model.decode(c_code)
show_segmentation(torch.softmax(segmentation_rec, dim=1), "loaded_segmentation_codes.png")

def show_image(s, save_path="tmp.png"):
  s = s.detach().cpu().numpy().transpose(0,2,3,1)[0]
  s = ((s+1.0)*127.5).clip(0,255).astype(np.uint8)
  s = Image.fromarray(s)
  # display(s)
  s.save(save_path)

codebook_size = config.model.params.first_stage_config.params.embed_dim
z_indices_shape = c_indices.shape
z_code_shape = c_code.shape
z_indices = torch.randint(codebook_size, z_indices_shape, device=model.device)
x_sample = model.decode_to_img(z_indices, z_code_shape)
show_image(x_sample, "random_codes.png")

idx = z_indices
idx = idx.reshape(z_code_shape[0], z_code_shape[2], z_code_shape[3])

cidx = c_indices
cidx = cidx.reshape(c_code.shape[0], c_code.shape[2], c_code.shape[3])

temperature = 1.0
top_k = 100
update_every = 50

start_t = time.time()
for i in range(0, z_code_shape[2] - 0):
    if i <= 8:
        local_i = i
    elif z_code_shape[2] - i < 8:
        local_i = 16 - (z_code_shape[2] - i)
    else:
        local_i = 8
    for j in range(0, z_code_shape[3] - 0):
        if j <= 8:
            local_j = j
        elif z_code_shape[3] - j < 8:
            local_j = 16 - (z_code_shape[3] - j)
        else:
            local_j = 8

        i_start = i - local_i
        i_end = i_start + 16
        j_start = j - local_j
        j_end = j_start + 16

        patch = idx[:, i_start:i_end, j_start:j_end]
        patch = patch.reshape(patch.shape[0], -1)
        cpatch = cidx[:, i_start:i_end, j_start:j_end]
        cpatch = cpatch.reshape(cpatch.shape[0], -1)
        patch = torch.cat((cpatch, patch), dim=1)
        logits, _ = model.transformer(patch[:, :-1])
        logits = logits[:, -256:, :]
        logits = logits.reshape(z_code_shape[0], 16, 16, -1)
        logits = logits[:, local_i, local_j, :]

        logits = logits / temperature

        if top_k is not None:
            logits = model.top_k_logits(logits, top_k)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx[:, i, j] = torch.multinomial(probs, num_samples=1)

        step = i * z_code_shape[3] + j
        if step % update_every == 0 or step == z_code_shape[2] * z_code_shape[3] - 1:
            x_sample = model.decode_to_img(idx, z_code_shape)
            print(f"Time: {time.time() - start_t} seconds")
            print(f"Step: ({i},{j}) | Local: ({local_i},{local_j}) | Crop: ({i_start}:{i_end},{j_start}:{j_end})")
            show_image(x_sample, f"sample_step{step}.png")
