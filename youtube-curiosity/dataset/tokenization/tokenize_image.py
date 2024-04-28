import torch
from PIL import Image
import torchvision.transforms as transforms
from diffusers import AmusedPipeline
import os

# LWM used the VQGAN encoder from aMUSEd to tokenize images
pipe = AmusedPipeline.from_pretrained(
    "amused/amused-256", variant="fp16", torch_dtype=torch.float16
)
pipe.vqvae.to(torch.float32)  # vqvae is producing nans in fp16
pipe = pipe.to("cuda")

vq = pipe.vqvae


def tokenize_image(video_id, frame_number, data_dir_path):
    """
    Use the VQGAN encoder from aMUSEd to tokenize an image
    """
    #tokenize an image
    image_path = f"{data_dir_path}videos/{video_id}/frames/{frame_number}-middle.png"

    if not os.path.exists(image_path):
        return None

    image = Image.open(image_path)
    image = transforms.Resize((256, 256))(image)
    image = transforms.ToTensor()(image)
    image = image.to(torch.float32)
    image = image.to("cuda")

    encoded = vq.encode(image.unsqueeze(0).to("cuda")).latents

    _, _, qtz = vq.quantize(encoded)
    qtz = qtz[2]

    return qtz

if __name__ == "__main__":
    tokenize_image("9SyEYd7X-fQ", 0, data_dir_path="/grogu/user/mhzhou/youtube-curiosity/dataset")
