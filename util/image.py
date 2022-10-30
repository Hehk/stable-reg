import torch
from torchvision import transforms as tfms
from matplotlib import pyplot as plt

def image_to_latents(image, vae, torch_device="cuda"):
  input_image = image.resize((512, 512))
  with torch.no_grad():
    latents = vae.encode(tfms.ToTensor()(input_image).unsqueeze(0).to(torch_device)*2-1)
  return 0.18215 * latents.latent_dist.sample()

# helper to see what the latents look like
def view_latents(latents):
  _, axs = plt.subplot(1, 4, figsize=(16,4)) 
  for channel in range(4):
    axs[channel].imshow(latents[0][channel].cpu(), cmap="Greys")