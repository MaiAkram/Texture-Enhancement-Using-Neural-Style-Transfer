import torch
from torch import nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
from PIL import Image
from tqdm.autonotebook import tqdm
import os
from torchmetrics.functional import structural_similarity_index_measure as ssim

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.chosen_features = ['0', '5', '10', '19', '28']
        self.model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:29]

    def forward(self, x):
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.chosen_features:
                features.append(x)
        return features

device = "cuda" if torch.cuda.is_available() else "cpu"
image_size = 356

image_loader = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

def load_image(image_name):
    image = Image.open(image_name)
    image = image_loader(image).unsqueeze(0)
    return image.to(device)

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).to(tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).to(tensor.device).view(1, 3, 1, 1)
    return tensor * std + mean

# Paths to folders
styl_path = 'E:/UNI/MS/Advanced Digital Image and Video Processing/Project/Code/Texture_Enhancement_Using_Neural_Style_Transfer/Style'
or_path = 'E:/UNI/MS/Advanced Digital Image and Video Processing/Project/Code/Texture_Enhancement_Using_Neural_Style_Transfer/Content'
output_path = 'E:/UNI/MS/Advanced Digital Image and Video Processing/Project/Code/Texture_Enhancement_Using_Neural_Style_Transfer/Output'

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Load images
style_img = []
styl_classNames = []
original_img = []
or_classNames = []
styl_list = os.listdir(styl_path)
or_list = os.listdir(or_path)

for cl in styl_list:
    if cl.lower().endswith(('.png', '.jpg', '.jpeg')):
        curImg = load_image(os.path.join(styl_path, cl))
        style_img.append(curImg)
        styl_classNames.append(os.path.splitext(cl)[0])

for cl in or_list:
    if cl.lower().endswith(('.png', '.jpg', '.jpeg')):
        curImg = load_image(os.path.join(or_path, cl))
        original_img.append(curImg)
        or_classNames.append(os.path.splitext(cl)[0])

# Create model
model = VGG().to(device).eval()

# Hyperparameters
iterations = 6000
learning_rate = 0.001
alpha = 1
beta = 0.01

# Compute style features and Gram matrices for all style images
style_features = []
gram_style = []
with torch.inference_mode():
    for img in style_img:
        features = model(img)
        style_features.append(features)
        gram_style_per_image = []
        for style_feature in features:
            batch_size, channel, height, width = style_feature.shape
            GS = style_feature.view(channel, height * width).mm(
                style_feature.view(channel, height * width).t()
            )
            gram_style_per_image.append(GS)
        gram_style.append(gram_style_per_image)

# Process each original image with each style image
for orig_idx, (orig_img, orig_name) in enumerate(zip(original_img, or_classNames)):
    # Compute content features for the original image
    with torch.inference_mode():
        original_image_features = model(orig_img)
        orig_content = original_image_features[4]  # conv5_1

    for style_idx, (style_name, style_gram) in enumerate(zip(styl_classNames, gram_style)):
        # Initialize generated image and optimizer
        generated = orig_img.clone().requires_grad_(True)
        optimizer = optim.Adam([generated], lr=learning_rate)

        # Training loop
        for iteration in tqdm(range(iterations), desc=f"Processing {orig_name} with style {style_name}"):
            generated_features = model(generated)
            gen_content = generated_features[4]  # conv5_1
            original_loss = torch.mean((gen_content - orig_content) ** 2)
            style_loss = 0

            for gen_feature, GS in zip(generated_features, style_gram):
                batch_size, channel, height, width = gen_feature.shape
                GG = gen_feature.view(channel, height * width).mm(
                    gen_feature.view(channel, height * width).t()
                )
                style_loss += torch.mean((GG - GS) ** 2)

            total_loss = alpha * original_loss + beta * style_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Save intermediate results
            if iteration % 2000 == 0:
                print(f"Iteration {iteration}, Total Loss: {total_loss.item():.4f}")
                output_filename = f"{orig_name}_styled_{style_name}_iter{iteration}.png"
                save_image(denormalize(generated), os.path.join(output_path, output_filename))

        # Save final result
        output_filename = f"{orig_name}_styled_{style_name}_final.png"
        save_image(denormalize(generated), os.path.join(output_path, output_filename))

        # Calculate and print SSIM
        similarity = ssim(denormalize(orig_img), denormalize(generated))
        print(f"SSIM for {orig_name} with style {style_name}: {similarity.item():.2f}")