"""
=========================================================
UAIRE

Image Preprocessor
=========================================================
"""

from PIL import Image

from torchvision import transforms


class ImagePreprocessor:

    def __init__(self):

        self.transform = transforms.Compose([

            transforms.Resize((32, 32)),

            transforms.ToTensor(),

            transforms.Normalize(

                (0.4914, 0.4822, 0.4465),

                (0.2023, 0.1994, 0.2010)

            )

        ])

    def process(self, image):

        if isinstance(image, str):

            image = Image.open(image).convert("RGB")

        tensor = self.transform(image)

        return tensor.unsqueeze(0)