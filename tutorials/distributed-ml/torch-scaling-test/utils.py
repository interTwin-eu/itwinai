from torchvision import datasets, transforms


def imagenet_dataset(data_root: str):
    """Create a torch dataset object for Imagenet."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=45),
        transforms.ColorJitter(
            brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    imagenet = datasets.ImageFolder(
        root=data_root,
        transform=transform
    )
    return imagenet
