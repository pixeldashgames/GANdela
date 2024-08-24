import torch

# !pip install --upgrade albumentations
from utils import save_checkpoint, load_checkpoint, save_some_examples
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


def train(
    netG: Generator,
    netD: Discriminator,
    train_dl,
    OptimizerG: optim.Adam,
    OptimizerD: optim.Adam,
    gen_loss,
    dis_loss,
    step_ahead=0,
):
    # loop = tqdm(train_dl, dynamic_ncols=True)
    for x, y in train_dl:
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)
        y = y.permute(0, 3, 1, 2).float()
        # Train Discriminator
        y_fake = netG(x).float()
        d_real = netD(x, y)
        d_real_loss = dis_loss(d_real, torch.ones_like(d_real))
        d_fake = netD(x, y_fake.detach())
        d_fake_loss = dis_loss(d_fake, torch.zeros_like(d_fake))
        d_loss = (d_real_loss + d_fake_loss) / 2

        netD.zero_grad()
        Dis_loss.append(d_loss.item())
        d_loss.backward()
        OptimizerD.step()

        # Train Generator
        d_fake = netD(x, y_fake)
        g_fake_loss = gen_loss(d_fake, torch.ones_like(d_fake))
        loss = gen_loss(y_fake, y)  # * config.L1_LAMBDA
        g_loss = (g_fake_loss + loss) / 2

        Gen_loss.append(g_loss.item())
        g_loss.backward()
        OptimizerG.step()

        for _ in range(step_ahead):
            OptimizerG.zero_grad()
            y_fake = netG(x).float()
            d_fake = netD(x, y_fake)
            g_fake_loss = gen_loss(d_fake, torch.ones_like(d_fake))
            loss = gen_loss(y_fake, y)  # * config.L1_LAMBDA

            g_loss = (g_fake_loss + loss) / 2
            Gen_loss.append(g_loss.item())
            g_loss.backward()
            OptimizerG.step()

    #     if idx % 10 == 0:
    #         loop.set_postfix(
    #             d_real=torch.sigmoid(d_real).mean().item(),
    #             d_fake=torch.sigmoid(d_fake).mean().item(),
    #         )
    # print("d_real: " + str(torch.sigmoid(d_real).mean().item()))
    # print("d_fake: " + str(torch.sigmoid(d_fake).mean().item()))


def main():
    start = 0
    netD = Discriminator(in_channels=3).to(config.DEVICE)
    netG = Generator(in_channels=3).to(config.DEVICE)
    optimizerD = torch.optim.Adam(
        netD.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, 0.999)
    )
    optimizerG = torch.optim.Adam(
        netG.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, 0.999)
    )
    dis_loss = nn.MSELoss()
    gen_loss = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN, start - 1, netG, optimizerG, config.LEARNING_RATE
        )
        load_checkpoint(
            config.CHECKPOINT_DISC, start - 1, netD, optimizerD, config.LEARNING_RATE
        )

    train_dataset = Satellite2Map_Data(root=config.TRAIN_DIR)
    train_dl = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_dataset = Satellite2Map_Data(root=config.VAL_DIR)
    val_dl = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    for epoch in range(start, config.NUM_EPOCHS):
        train(netG, netD, train_dl, optimizerG, optimizerD, gen_loss, dis_loss)
        if config.SAVE_MODEL and epoch % 100 == 0 and epoch > 0:
            save_checkpoint(
                netG,
                optimizerG,
                epoch,
                Gen_loss[-1],
                Dis_loss[-1],
                filename=f"./checkpoints/{epoch}_{config.CHECKPOINT_GEN}",
            )
            save_checkpoint(
                netD,
                optimizerD,
                epoch,
                Gen_loss[-1],
                Dis_loss[-1],
                filename=f"./checkpoints/{epoch}_{config.CHECKPOINT_DISC}",
            )
        if epoch % 100 == 0:
            print("save example")
            try:
                save_some_examples(netG, val_dl, epoch, folder="evaluation")
            except Exception as e:
                print(f"Something went wrong with epoch {epoch}: {e}")

        print(
            "Epoch :", epoch, " Gen Loss :", Gen_loss[-1], "Disc Loss :", Dis_loss[-1]
        )
    save_checkpoint(
        netG,
        optimizerG,
        "final",
        Gen_loss[-1],
        Dis_loss[-1],
        filename=f"./checkpoints/{epoch}_{config.CHECKPOINT_GEN}",
    )
    save_checkpoint(
        netD,
        optimizerD,
        "final",
        Gen_loss[-1],
        Dis_loss[-1],
        filename=f"./checkpoints/{epoch}_{config.CHECKPOINT_DISC}",
    )
    print("save example")
    try:
        save_some_examples(netG, val_dl, 400, folder="evaluation")
    except Exception as e:
        print(f"Something went wrong with the last epoch")


main()

# if __name__ == '__main__':
#     main()
