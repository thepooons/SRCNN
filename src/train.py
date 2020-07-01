import os
from tqdm import tqdm
import torch
import matplotlib.image as img_plt

def train_loop(
    epochs,
    model,
    device,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    test_loader,
    enhanced_image_path,
    verbose
):
    os.system('mkdir ' + enhanced_image_path)
    os.system('mkdir models')
    model = model.to(device)
    criterion = criterion.to(device)
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0

        model.train()
        for x, y in train_loader:
            # fetch the features and targets from train_loader
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            z = model(x)
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            torch.cuda.empty_cache()

        # print validation image and model predictions 
        verbose = int(verbose)
        if epoch%verbose == 0:
            with torch.no_grad():
                model.eval()
                
                for image, image_name in test_loader:
                    image = image.to(device)                         # LR
                    big_image = model(image).detach().cpu().numpy()  # HR
                    img_plt.imsave(enhanced_image_path + '/enh_' + str(image_name[0])[:-4] + '_' + str(epoch) + '.png', big_image.squeeze().squeeze(), cmap='Greys_r')              
                print('=' * 20, 'EPOCH LOSS', epoch_loss, '=' * 20)
                torch.save(model.state_dict(), './models/ckpt_' + str(epoch) + '.pth')  