from .models import SRCNN
from .datasets import T91Set
import torch

model = SRCNN(
    c=1,
    n1=64,
    n2=32,
    n3=1,
    f1=9,
    f2=3,
    f3=5,
)

def create_datasets(train_img_path, test_img_path, val_img_path):
    train_set = T91Set(
        image_dir=train_img_path,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=4,
        shuffle=True,
    )

    val_set = T91Set(
    image_dir=val_img_path,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=1,
        shuffle=True,
    )

    test_set = T91Set(
        image_dir=test_img_path,
        isTest=True,
    )


    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=True,
    )

    return train_loader, val_loader, test_loader