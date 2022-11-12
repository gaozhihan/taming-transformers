import sys
import os
from omegaconf import OmegaConf
import yaml
from PIL import Image
import numpy as np
import torch
import time
from taming.models.vqgan import VQModel, GumbelVQ


# download a VQGAN with f=16 (16x compression per spatial dimension) and with a codebook with 1024 entries
os.makedirs("logs/vqgan_imagenet_f16_1024/checkpoints", exist_ok=True)
os.system("wget 'https://heibox.uni-heidelberg.de/f/140747ba53464f49b476/?dl=1' -O 'logs/vqgan_imagenet_f16_1024/checkpoints/last.ckpt'")
os.makedirs("logs/vqgan_imagenet_f16_1024/configs", exist_ok=True)
os.system("wget 'https://heibox.uni-heidelberg.de/f/6ecf2af6c658432c8298/?dl=1' -O 'logs/vqgan_imagenet_f16_1024/configs/model.yaml'")
# download a VQGAN with f=16 (16x compression per spatial dimension) and with a larger codebook (16384 entries)
os.makedirs("logs/vqgan_imagenet_f16_16384/checkpoints", exist_ok=True)
os.system("wget 'https://heibox.uni-heidelberg.de/f/867b05fc8c4841768640/?dl=1' -O 'logs/vqgan_imagenet_f16_16384/checkpoints/last.ckpt'")
os.makedirs("logs/vqgan_imagenet_f16_16384/configs", exist_ok=True)
os.system("wget 'https://heibox.uni-heidelberg.de/f/274fb24ed38341bfa753/?dl=1' -O 'logs/vqgan_imagenet_f16_16384/configs/model.yaml'")
# download a VQGAN with f=8 (8x compression per spatial dimension) and a larger codebook-size with 8192 entries
os.makedirs("logs/vqgan_gumbel_f8/checkpoints", exist_ok=True)
os.system("wget 'https://heibox.uni-heidelberg.de/f/34a747d5765840b5a99d/?dl=1' -O 'logs/vqgan_gumbel_f8/checkpoints/last.ckpt'")
os.makedirs("logs/vqgan_gumbel_f8/configs", exist_ok=True)
os.system("wget 'https://heibox.uni-heidelberg.de/f/b24d14998a8d4f19a34f/?dl=1' -O 'logs/vqgan_gumbel_f8/configs/model.yaml'")

torch.set_grad_enabled(False)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config

def load_vqgan(config, ckpt_path=None, is_gumbel=False):
  if is_gumbel:
    model = GumbelVQ(**config.model.params)
  else:
    model = VQModel(**config.model.params)
  if ckpt_path is not None:
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
  return model.eval()

def preprocess_vqgan(x):
  x = 2.*x - 1.
  return x

def custom_to_pil(x):
  x = x.detach().cpu()
  x = torch.clamp(x, -1., 1.)
  x = (x + 1.)/2.
  x = x.permute(1,2,0).numpy()
  x = (255*x).astype(np.uint8)
  x = Image.fromarray(x)
  if not x.mode == "RGB":
    x = x.convert("RGB")
  return x

def reconstruct_with_vqgan(x, model):
  # could also use model(x) for reconstruction but use explicit encoding and decoding here
  z, _, [_, _, indices] = model.encode(x)
  print(f"VQGAN --- {model.__class__.__name__}: latent shape: {z.shape[2:]}")
  xrec = model.decode(z)
  return xrec

config1024 = load_config("logs/vqgan_imagenet_f16_1024/configs/model.yaml", display=False)
config16384 = load_config("logs/vqgan_imagenet_f16_16384/configs/model.yaml", display=False)

model1024 = load_vqgan(config1024, ckpt_path="logs/vqgan_imagenet_f16_1024/checkpoints/last.ckpt").to(DEVICE)
model16384 = load_vqgan(config16384, ckpt_path="logs/vqgan_imagenet_f16_16384/checkpoints/last.ckpt").to(DEVICE)

config32x32 = load_config("logs/vqgan_gumbel_f8/configs/model.yaml", display=False)
model32x32 = load_vqgan(config32x32, ckpt_path="logs/vqgan_gumbel_f8/checkpoints/last.ckpt", is_gumbel=True).to(DEVICE)

titles = ["Input", "VQGAN (f8, 8192)",
          "VQGAN (f16, 16384)", "VQGAN (f16, 1024)"]

import io
import requests
import PIL
from PIL import Image
from PIL import ImageDraw, ImageFont
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from dall_e import map_pixels, unmap_pixels, load_model

font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf", 22)

def download_image(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return PIL.Image.open(io.BytesIO(resp.content))


def preprocess(img, target_image_size=256, map_dalle=True):
    s = min(img.size)

    if s < target_image_size:
        raise ValueError(f'min dim for image {s} < {target_image_size}')

    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    if map_dalle:
        img = map_pixels(img)
    return img

def reconstruct_with_dalle(x, encoder, decoder, do_preprocess=False):
    # takes in tensor (or optionally, a PIL image) and returns a PIL image
    if do_preprocess:
        x = preprocess(x)
    z_logits = encoder(x)
    z = torch.argmax(z_logits, axis=1)

    print(f"DALL-E: latent shape: {z.shape}")
    z = F.one_hot(z, num_classes=encoder.vocab_size).permute(0, 3, 1, 2).float()

    x_stats = decoder(z).float()
    x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))
    x_rec = T.ToPILImage(mode='RGB')(x_rec[0])

    return x_rec

def stack_reconstructions(input, x0, x1, x2, titles=[]):
    assert input.size == x1.size == x2.size
    w, h = input.size[0], input.size[1]
    img = Image.new("RGB", (5 * w, h))
    img.paste(input, (0, 0))
    img.paste(x0, (1 * w, 0))
    img.paste(x1, (2 * w, 0))
    img.paste(x2, (3 * w, 0))
    for i, title in enumerate(titles):
        ImageDraw.Draw(img).text((i * w, 0), f'{title}', (255, 255, 255), font=font)  # coordinates, text, color, font
    return img

def reconstruction_pipeline(url, size=320):
    x_vqgan = preprocess(download_image(url), target_image_size=size, map_dalle=False)
    x_vqgan = x_vqgan.to(DEVICE)

    print(f"input is of size: {x_vqgan.shape}")
    x0 = reconstruct_with_vqgan(preprocess_vqgan(x_vqgan), model32x32)
    x1 = reconstruct_with_vqgan(preprocess_vqgan(x_vqgan), model16384)
    x2 = reconstruct_with_vqgan(preprocess_vqgan(x_vqgan), model1024)
    img = stack_reconstructions(custom_to_pil(preprocess_vqgan(x_vqgan[0])),
                                custom_to_pil(x0[0]), custom_to_pil(x1[0]),
                                custom_to_pil(x2[0]), titles=titles)
    return img

img = reconstruction_pipeline(url='https://heibox.uni-heidelberg.de/f/7bb608381aae4539ba7a/?dl=1', size=384)
img.save("tmp_img1.png")
img = reconstruction_pipeline(url='https://heibox.uni-heidelberg.de/f/6f12b330eb564d288d76/?dl=1', size=384)
img.save("tmp_img2.png")
img = reconstruction_pipeline(url='https://heibox.uni-heidelberg.de/f/8555a959b0a5423cbfd1/?dl=1', size=384)
img.save("tmp_img3.png")
img = reconstruction_pipeline(url='https://heibox.uni-heidelberg.de/f/be6f4ff34e1544109563/?dl=1', size=384)
img.save("tmp_img4.png")
img = reconstruction_pipeline("https://heibox.uni-heidelberg.de/f/e41f5053cbd34f11a8d5/?dl=1", size=384)
img.save("tmp_img5.png")
img = reconstruction_pipeline(url='https://assets.bwbx.io/images/users/iqjWHBFdfxIU/iKIWgaiJUtss/v2/1000x-1.jpg', size=384)
img.save("tmp_img6.png")
