from torch.utils.data import Dataset
import random
import torchvision.transforms as transforms
from PIL import Image
from main import IMAGE_SIZE

class DummyDataset(Dataset):
    def __init__(self, num_samples, image_size=(IMAGE_SIZE, IMAGE_SIZE, 3), transform=None):
        self.num_samples = num_samples
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate a random PIL Image (e.g., RGB with random pixel values)
        # For simplicity, we'll create a black image and pretend it's data
        # In a real scenario, this would load an actual image
        img_width, img_height, img_channels = self.image_size[0], self.image_size[1], self.image_size[2]
        random_array = bytearray([random.randint(0, 255) for _ in range(img_width * img_height * img_channels)])
        image = Image.frombytes('RGB', (img_width, img_height), bytes(random_array))

        if self.transform:
            context_view, target_view = self.transform(image)
            return context_view, target_view

        # If no transform, return the raw image (though our setup expects transform)
        return transforms.ToTensor()(image)