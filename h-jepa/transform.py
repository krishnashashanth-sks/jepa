class HJEPADatasetTransform:
    def __init__(self, weak_augment, strong_augment, patch_masking):
        self.weak_augment = weak_augment
        self.strong_augment = strong_augment
        self.patch_masking = patch_masking

    def __call__(self, x):
        # x is assumed to be a PIL Image for torchvision transforms

        # Create context view: weak augmentation + masking
        context_view_raw = self.weak_augment(x)
        context_view = self.patch_masking(context_view_raw) # Mask the weakly augmented view

        # Create target view: strong augmentation
        target_view = self.strong_augment(x)

        return context_view, target_view

print("HJEPADatasetTransform class defined, ready to combine masking and augmentation.")