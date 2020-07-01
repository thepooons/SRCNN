from .datasets import T91Set
from .dispatcher import model, create_datasets
from .train import train_loop
import torch
import torch.nn as nn

if __name__ == '__main__':
    train_loader, val_loader, test_loader = create_datasets(
        train_img_path=r't91-image-data\sris_srcnn_data\T91',
        val_img_path=r't91-image-data\sris_srcnn_data\T91_val',
        test_img_path=r't91-image-data\sris_srcnn_data\T91_test',
    )
    
    optimizer = torch.optim.Adam(model.parameters())

    train_loop(
        epochs=200,
        model=model,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=nn.MSELoss(),
        optimizer=optimizer,
        enhanced_image_path='enhanced_images',
        verbose=50,
    )