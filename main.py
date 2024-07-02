import torch
import torchvision
from torchvision.datasets import ImageFolder

from diffusion_model import DiffusionModel
from generate_images import generate_images
from train import train_diffusion_model
from display import display

IMAGE_SIZE = 64
BATCH_SIZE = 64
DATASET_REPETITIONS = 5
NOISE_EMBEDDING_SIZE = 32

EMA = 0.999
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 100
NUM_DIFFUSION_STEPS = 20

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FIXED_GENERATOR = torch.Generator().manual_seed(42)

transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        torchvision.transforms.ToTensor(),
    ]
)

train_dataset = ImageFolder("data/flower-dataset/dataset/train", transforms)

trainloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=FIXED_GENERATOR,
    pin_memory=True,
    num_workers=4,
    drop_last=True,
)

images = next(iter(trainloader))[0][:10, ...]
images = torch.transpose(images, 1, 3)
display(images.numpy(), save_to="original_images.png")

mean = torch.zeros(3).to(device)
std = torch.zeros(3).to(device)

for _ in range(DATASET_REPETITIONS):
    for idx, batch in enumerate(trainloader):
        image = batch[0].to(device)
        image_mean = torch.mean(image, dim=(0, 2, 3))
        image_std = torch.std(image, dim=(0, 2, 3))
        mean = torch.add(mean, image_mean)
        std = torch.add(std, image_std)

mean = (mean / (len(trainloader) * DATASET_REPETITIONS)).to("cpu")
std = (std / (len(trainloader) * DATASET_REPETITIONS)).to("cpu")

print(mean)
print(std)

# mean = torch.Tensor([0.4353, 0.3773, 0.2871]).to("cpu")
# std = torch.Tensor([0.2526, 0.1980, 0.2044]).to("cpu")

train_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std),
    ]
)

train_dataset = ImageFolder(
    "data/flower-dataset/dataset/train", train_transforms
)
trainloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    generator=FIXED_GENERATOR,
    num_workers=4,
    drop_last=True,
)

diffusion_model = DiffusionModel(
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    ema_value=EMA,
    noise_embedding_size=NOISE_EMBEDDING_SIZE,
    mean=mean,
    std=std,
    device=device,
)

optimizer = torch.optim.AdamW(
    params=filter(
        lambda param: param.requires_grad, diffusion_model.model.parameters()
    ),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

train_diffusion_model(
    model=diffusion_model,
    num_epochs=EPOCHS,
    optimizer=optimizer,
    loss_function=torch.nn.L1Loss(),
    trainloader=trainloader,
    device=device,
    image_generator_function=generate_images,
    dataset_repetitions=DATASET_REPETITIONS,
    num_images_to_generate=10,
    num_generate_diffusion_steps=NUM_DIFFUSION_STEPS,
)

generate_images(
    model=diffusion_model,
    num_images=10,
    num_diffusion_steps=NUM_DIFFUSION_STEPS,
    filename="final_generated_images.png",
)
