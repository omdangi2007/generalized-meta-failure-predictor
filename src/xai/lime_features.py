"""
=========================================================
UAIRE

LIME Image Explainer

Explains which image superpixels influenced the backbone
prediction without modifying the backbone architecture.
=========================================================
"""

import numpy as np
import torch
from PIL import Image
from skimage.segmentation import mark_boundaries, slic
from torchvision import transforms


class LIMEImageExplainer:

    """
    Generates LIME superpixel explanations for a PyTorch
    image classification backbone.
    """

    def __init__(self, model, device):

        from lime import lime_image

        self.model = model
        self.device = device
        self.explainer = lime_image.LimeImageExplainer(
            random_state=42
        )
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010),
                ),
            ]
        )

    def explain(
        self,
        image,
        class_index=None,
        num_samples=384,
        num_features=8,
        batch_size=32,
        max_side=160,
    ):

        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        original_size = image.size
        lime_image = image.copy()
        lime_image.thumbnail(
            (max_side, max_side),
            Image.Resampling.BILINEAR,
        )
        image_array = np.asarray(lime_image).astype(np.double) / 255.0

        if class_index is None:
            probabilities = self._predict_batch(
                [image_array],
                batch_size=batch_size,
            )
            class_index = int(np.argmax(probabilities[0]))
            confidence = float(probabilities[0, class_index])
        else:
            probabilities = self._predict_batch(
                [image_array],
                batch_size=batch_size,
            )
            confidence = float(probabilities[0, class_index])

        explanation = self.explainer.explain_instance(
            image_array,
            classifier_fn=lambda images: self._predict_batch(
                images,
                batch_size=batch_size,
            ),
            labels=(class_index,),
            hide_color=0,
            num_samples=num_samples,
            batch_size=batch_size,
            segmentation_fn=lambda x: slic(
                x,
                n_segments=48,
                compactness=12,
                sigma=1,
                start_label=0,
            ),
        )

        positive_image, positive_mask = explanation.get_image_and_mask(
            class_index,
            positive_only=True,
            num_features=num_features,
            hide_rest=False,
        )
        signed_image, signed_mask = explanation.get_image_and_mask(
            class_index,
            positive_only=False,
            num_features=num_features,
            hide_rest=False,
        )

        positive_overlay = mark_boundaries(
            positive_image,
            positive_mask,
            color=(0.1, 1.0, 0.55),
            mode="thick",
        )
        signed_overlay = mark_boundaries(
            signed_image,
            signed_mask,
            color=(1.0, 0.82, 0.12),
            mode="thick",
        )

        local_explanation = explanation.local_exp.get(
            class_index,
            [],
        )
        top_superpixels = [
            {
                "superpixel": int(superpixel),
                "weight": float(weight),
                "direction": (
                    "Supports prediction"
                    if weight >= 0
                    else "Contradicts prediction"
                ),
            }
            for superpixel, weight in sorted(
                local_explanation,
                key=lambda item: abs(item[1]),
                reverse=True,
            )[:num_features]
        ]

        return {
            "class_index": class_index,
            "confidence": confidence,
            "positive_overlay": self._resize_to_original(
                self._to_uint8(positive_overlay),
                original_size,
            ),
            "signed_overlay": self._resize_to_original(
                self._to_uint8(signed_overlay),
                original_size,
            ),
            "positive_mask": positive_mask.astype(np.int32),
            "signed_mask": signed_mask.astype(np.int32),
            "segments": explanation.segments.astype(np.int32),
            "top_superpixels": top_superpixels,
        }

    def _predict_batch(self, images, batch_size=32):

        tensors = []

        for image in images:
            image_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
            tensors.append(self.transform(image_uint8))

        probabilities = []

        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(tensors), batch_size):
                batch = torch.stack(
                    tensors[start:start + batch_size]
                ).to(self.device)
                logits = self.model(batch)
                probs = torch.softmax(logits, dim=1)
                probabilities.append(probs.detach().cpu().numpy())

        return np.concatenate(probabilities, axis=0)

    def _to_uint8(self, image):

        return np.clip(image * 255.0, 0, 255).astype(np.uint8)

    def _resize_to_original(self, image, original_size):

        return np.asarray(
            Image.fromarray(image).resize(
                original_size,
                Image.Resampling.BILINEAR,
            )
        )
