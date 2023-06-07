import os
try:
    os.chdir('code')
except FileNotFoundError:
    pass


from utils import set_seed, BPRLoss, getFileName
from world import Config, cprint
import torch
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
from model import PureMF, LightGCN
from register import print_config_info, load_dataset


def run_training():
    # Instantiate the config object
    config = Config()
    # Print config information
    print_config_info(config)
    # Set the seed
    set_seed(config.seed)
    print(">>SEED:", config.seed)
    # Load the data
    dataset = load_dataset(config)
    # Create a model string to model map
    models = {
        'mf': PureMF,
        'lgn': LightGCN
    }
    # Instantiate the recommender system model
    Recmodel = models[config.model_name](config, dataset)
    # Move the model to the device
    Recmodel = Recmodel.to(config.device)
    # Instantiate the BPRLoss
    bpr = BPRLoss(Recmodel, config)

    weight_file = getFileName(config)
    print(f"load and save to {weight_file}")
    if config.load_bool:
        try:
            Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
            cprint(f"loaded model weights from {weight_file}")
        except FileNotFoundError:
            print(f"{weight_file} not exists, start from beginning")
    # Instantiate the procedure manager
    procedure_manager = Procedure.ProcedureManager(config)

    # init tensorboard
    if config.tensorboard:
        w : SummaryWriter = SummaryWriter(
            join(config.board_path, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + config.comment)
        )
    else:
        w = None
        cprint("not enable tensorflowboard")

    try:
        for epoch in range(config.train_epochs):
            start = time.time()
            if epoch % 10 == 0:
                cprint("[TEST]")
                if epoch % 50 == 0:
                    print('t')
                procedure_manager.test(dataset, Recmodel, epoch, w, config.multicore)
            output_information = procedure_manager.bpr_train_original(
                dataset, Recmodel, bpr, epoch, w=w, config=config
            )
            print(f'EPOCH[{epoch+1}/{config.train_epochs}] {output_information}')
            torch.save(Recmodel.state_dict(), weight_file)
    finally:
        if config.tensorboard:
            w.close()


if __name__ == '__main__':
    run_training()
