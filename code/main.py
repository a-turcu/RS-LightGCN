import world
import utils
from model import PureMF, LightGCN
from world import cprint
import torch
from tensorboardX import SummaryWriter
import time
import procedure
from os.path import join
from register import print_config_info, load_dataset


def run_training():
    # Instantiate the config object
    config = world.Config()
    # Print config information
    print_config_info(config)
    # Set the seed
    utils.set_seed(config.seed)
    print(">>SEED:", config.seed)
    # Load the data
    dataset = load_dataset(config)
    # Create a model string to model map
    models = {
        'mf': PureMF,
        'lgn': LightGCN
    }
    # Instantiate the recommender system model
    rec_model = models[config.model_name](config, dataset)
    # Move the model to the device
    rec_model = rec_model.to(config.device)
    # Instantiate the BPRLoss
    loss = utils.BrpLoss(rec_model, config)

    weight_file = utils.get_file_name(config)
    print(f"load and save to {weight_file}")
    if config.load_bool:
        try:
            rec_model.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
            cprint(f"loaded model weights from {weight_file}")
        except FileNotFoundError:
            print(f"{weight_file} not exists, start from beginning")
    # Instantiate the procedure manager
    procedure_manager = procedure.ProcedureManager(config)

    # init tensorboard
    if config.tensorboard:
        w: SummaryWriter = SummaryWriter(
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
                procedure_manager.test(dataset, rec_model, epoch, w, config.multicore)
            output_information = procedure_manager.bpr_train_original(
                dataset, rec_model, loss, epoch, w=w, config=config
            )
            print(f'EPOCH[{epoch+1}/{config.train_epochs}] {output_information}')
            torch.save(rec_model.state_dict(), weight_file)
    finally:
        if config.tensorboard:
            w.close()


if __name__ == '__main__':
    run_training()