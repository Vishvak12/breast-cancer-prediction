from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
def get_data_loaders(data_dir, train_ratio=0.8, batch_size=32, image_size=(224, 224), shuffle=True):
    """
    Splits the dataset into training and testing datasets and returns corresponding data loaders.

    Args:
    - data_dir (str): Path to the dataset directory, where each class is in a separate folder.
    - train_ratio (float): Proportion of the data to use for training (between 0 and 1).
    - batch_size (int): Batch size for the data loaders.
    - image_size (tuple): Image size for resizing.
    - shuffle (bool): Whether to shuffle the dataset before splitting.

    Returns:
    - train_loader (DataLoader): DataLoader for the training set.
    - test_loader (DataLoader): DataLoader for the testing set.
    """
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the dataset with ImageFolder
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Calculate split lengths
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    # Split the dataset
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
