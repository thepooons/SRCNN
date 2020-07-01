import os
from tqdm import tqdm
import torch

def train(
    epochs,
    model,
    device,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    test_loader
):
    os.system('mkdir models')
    os.system('mkdir big_goose_pics')
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
    #             print(z.dtype)
    #             print(y.dtype)
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            torch.cuda.empty_cache()

        # print validation image and model predictions 
        if epoch%50 == 0:
            with torch.no_grad():
                model.eval()
                
                for goose in test_loader:
                    goose = goose.to(device)                         # LR
                    big_goose = model(goose).detach().cpu().numpy()  # HR
                    print(big_goose.shape)
                    plt.imshow(big_goose.squeeze().squeeze(), cmap='Greys_r')
                    plt.show()
                print('=' * 20, 'EPOCH LOSS', epoch_loss, '=' * 20)
                torch.save(model.state_dict(), './models/ckpt_' + str(epoch) + '.pth')  
                img_plt.imsave('./big_goose_pics/big_goose_' + str(epoch) + '.png', big_goose.squeeze().squeeze(), cmap='Greys_r')              