import os
import logging
from _version import __version__
import hydra
import json
from omegaconf import OmegaConf
import warnings
from torch.autograd import Variable

import statistics
import torch

import shutil
import time


os.environ["CUDA_VISIBLE_DEVICES"] = "3"
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available. Training on CPU')
else:
    print('CUDA is available. Training on GPU')


import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from src.models.ltformer_model import ltformer_d128
from src.datasets.train_triplet_loader import TripletDataset
from src.losses.triplet_loss_layers import loss_factory
from src.utils.path import get_cwd
from src.utils.restart import restart_from_checkpoint

HYDRA_FULL_ERROR=1

writer = SummaryWriter()
logger = logging.getLogger("Training")
logger.setLevel(logging.INFO)
device = torch.device("cuda")
warnings.filterwarnings("ignore")

# print(torch.cuda.device_count())


@hydra.main(version_base=None, config_path="config", config_name="config_train_ltformer")
def main(cfg):

    logger.info("Version: " + __version__)
    dict_cfg = OmegaConf.to_container(cfg)
    cfg_pprint = json.dumps(dict_cfg, indent=4)
    logger.info(cfg_pprint)

    output_dir = get_cwd()
    logger.info(f"Working dir: {os.getcwd()}")
    logger.info(f"Export dir: {output_dir}")
    logger.info("Loading parameters from config file")
    data_dir_list = cfg.paths.train_data
    nb_epoch = cfg.params.nb_epoch
    batch_size = cfg.params.batch_size
    image_size = cfg.params.image_size
    initial_lr = cfg.params.lr
    margin_value = cfg.params.margin_value
    loss_weight = cfg.params.loss_weight

    # gathers all epoch losses 
    loss_list = []

    # creates dataset and datalaoder
    dataset = TripletDataset(data_dir_list, image_size)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=True, pin_memory=True)

    model = ltformer_d128()

    criterion = loss_factory(
        cfg.params.loss_layer,
        batch_size=batch_size,
        margin_value=margin_value,
        loss_weight=loss_weight,
    )

    optimizer = optim.SGD(
        model.parameters(),
        lr=initial_lr,
        momentum=cfg.params.momentum,
        weight_decay=cfg.params.weight_decay
    )

    # optionally resume from a checkpoint
    to_restore = {"epoch": 0}
    restart_from_checkpoint(
        os.path.join(output_dir, f"{cfg.params.model}.pth"),
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer
    )
    start_epoch = to_restore["epoch"]

    logger.info("Start Epochs ...")
    for epoch in range(start_epoch,nb_epoch):

        model.train()
        model.cuda()
        loss_epoch = []
        dist_positive_epoch = []
        dist_negative_epoch = []

        for (idx, data) in enumerate(train_loader):
            _, inputs = data
            input_var = torch.autograd.Variable(inputs.cuda())
            
            if not (list(input_var.size())[0] == batch_size):
                continue

            inputs_var_batch = input_var.view(batch_size * 3, 1, image_size, image_size)

            # computed output
            output = model(inputs_var_batch).to(device)
            # print(output.shape)

            dist_positive, dist_negative, loss = criterion(output)
            if len(dist_positive) == 0 and len(dist_negative) == 0:
                continue

            # save some metric values 
            loss_epoch.append(loss.item())
            dist_positive_epoch.append(dist_positive[0].item())
            dist_negative_epoch.append(dist_negative[0].item())

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print(prof)
        save_dict = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": loss
        }
        torch.save(
            save_dict,
            os.path.join(output_dir, f"{cfg.params.model}.pth"),
        )
        if epoch % cfg.params.checkpoint_freq == 0 or epoch == nb_epoch - 1:
            shutil.copyfile(
                os.path.join(output_dir, f"{cfg.params.model}.pth"),
                os.path.join(output_dir, f"{cfg.params.model}_" + str(epoch) + ".pth"),
            )
        loss_list.append(statistics.mean(loss_epoch))
        mean_dist_positive = statistics.mean(dist_positive_epoch)
        mean_dist_negative = statistics.mean(dist_negative_epoch)

        logger.info(f"Epoch= {epoch:04d}  Loss= {statistics.mean(loss_epoch):0.4f}\
        Mean-Dist-Pos: {mean_dist_positive:0.4f}\
        Mean-Dist-Neg: {mean_dist_negative:0.4f}")
        writer.add_scalar("loss", statistics.mean(loss_epoch), epoch)


    checkpoint = {"epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "loss": loss}
    checkpoint_export_path = os.path.join(output_dir, f"{cfg.params.model}.pth")
    torch.save(checkpoint, checkpoint_export_path)
    logger.info(f"Checkpoint savec to: {checkpoint_export_path}")


if __name__ == "__main__":
    main()
