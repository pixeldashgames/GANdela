import torch
from utils.utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
import config
from DataSet.dataset import Satellite2Map_Data
from pix2pix.Generator import Generator
from pix2pix.Discriminator import Discriminator
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm


torch.backends.cudnn.benchmark = True
Gen_loss = []
Dis_loss = []

def train(netG: Generator, netD: Discriminator, train_dl, OptimizerG: optim.Adam, OptimizerD: optim.Adam, L1_loss: nn.L1Loss, BCE_loss: nn.BCEWithLogitsLoss):
    loop = tqdm(train_dl, dynamic_ncols= True)
    for idx, (x,y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)
        y = y.permute(0,3,1,2)

        print(idx)
        # Train Discriminator
        y_fake = netG(x)
        d_real = netD(x,y)
        d_real_loss = BCE_loss(d_real, torch.ones_like(d_real))
        d_fake = netD(x,y_fake.detach())
        d_fake_loss = BCE_loss(d_fake, torch.zeros_like(d_fake))
        d_loss = (d_real_loss + d_fake_loss) / 2
        
        netD.zero_grad()
        Dis_loss.append(d_loss.item())
        d_loss.backward()
        OptimizerD.step()
        
        # Train Generator
        d_fake = netD(x,y_fake)
        g_fake_loss = BCE_loss(d_fake, torch.ones_like(d_fake))
        l1 = L1_loss(y_fake,y) * config.L1_LAMBDA
        g_loss = g_fake_loss + l1
        
        OptimizerG.zero_grad()
        Gen_loss.append(g_loss.item())
        g_loss.backward()
        OptimizerG.step()
        
        if idx % 10 == 0:
            loop.set_postfix(
                d_real = torch.sigmoid(d_real).mean().item(),
                d_fake = torch.sigmoid(d_fake).mean().item()
            )

def main():
    netD = Discriminator(in_channels=3).to(config.DEVICE)
    netG = Generator(in_channels=3).to(config.DEVICE)
    optimizerD = torch.optim.Adam(netD.parameters(), lr = config.LEARNING_RATE, betas=(config.BETA1, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr = config.LEARNING_RATE, betas=(config.BETA1, 0.999))
    bce_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()
    
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN, netG, optimizerG, config.LEARNING_RATE
        )
        load_checkpoint(
            config.CHECKPOINT_DISC, netD, optimizerD, config.LEARNING_RATE
        )
    
    train_dataset = Satellite2Map_Data(root=config.TRAIN_DIR)
    train_dl = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    val_dataset = Satellite2Map_Data(root=config.VAL_DIR)
    val_dl = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    
    for epoch in range(config.NUM_EPOCHS):
        train(
            netG, netD,train_dl,optimizerG,optimizerD,l1_loss,bce_loss
        )
        if config.SAVE_MODEL and epoch % 50 == 0:
            save_checkpoint(netG, optimizerG, filename=config.CHECKPOINT_GEN)
            save_checkpoint(netD, optimizerD, filename=config.CHECKPOINT_DISC)
        if epoch % 2 == 0:
           save_some_examples(netG,val_dl,epoch,folder="evaluation")
    save_checkpoint(netG, optimizerG, filename=config.CHECKPOINT_GEN)
    save_checkpoint(netD, optimizerD, filename=config.CHECKPOINT_DISC)


if __name__ == '__main__':
    main()