import pandas as pd

import utils
from model import PureMf, LightGCN
from parse import parse_args
from world import Config, cprint, FakeArgs
import torch
from tensorboardX import SummaryWriter
import time
import procedure
from os.path import join
from utils import print_config_info

from dataloader import DataLoader, LastfmLoader


def run_training(config: Config):
    # Print config information
    print_config_info(config)
    # Set the seed
    utils.set_seed(config.seed)
    print(">>SEED:", config.seed)
    # Load the data
    if config.dataset == 'lastfm':
        dataset = LastfmLoader(config)
    else:
        dataset = DataLoader(config)
    # Create a model string to model map
    models = {'mf': PureMf, 'lgn': LightGCN}
    # Instantiate the recommender system model
    rec_model = models[config.model_name](config, dataset)
    # Move the model to the device
    rec_model = rec_model.to(config.device)
    # Instantiate the BPRLoss
    loss = utils.BrpLoss(rec_model, config)
    # Load weight file
    weight_file = utils.get_checkpoint_file_name(config)
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
        data_list = []
        for epoch in range(config.train_epochs):
            start = time.time()
            if epoch % 10 == 0:
                cprint("[TEST]")
                results = procedure_manager.test(dataset, rec_model, epoch, w, config.multicore)
                if config.sampling == 'hard_neg':
                    procedure_manager.update_top_ranked_items(dataset, rec_model, config.multicore)
                elif config.sampling in ('hard_neg2', 'mixed'):
                    procedure_manager.update_top_ranked_items2(dataset, rec_model, config.multicore)
                elif config.sampling in ('hard_neg3', 'hard_neg4'):
                    procedure_manager.update_top_ranked_items3(dataset, rec_model, config.multicore)
                results = {k: float(v) for k, v in results.items()}
                results['epoch'] = epoch
                data_list.append(results)
                output_csv_results(data_list, config)

            output_information = procedure_manager.bpr_train_original(
                dataset, rec_model, loss, epoch, w=w, config=config
            )
            print(f'EPOCH[{epoch+1}/{config.train_epochs}] {output_information}')
            print(f'EPOCH[{epoch+1} took {int(time.time()-start)} seconds!')
            torch.save(rec_model.state_dict(), weight_file)
    finally:
        if config.tensorboard:
            w.close()


def output_csv_results(data_list, config):
    """
    Creates a csv output of the test results.
    """
    df = pd.DataFrame(data_list)
    new_column_order = ['epoch'] + [c for c in df.columns.tolist() if c != 'epoch']
    df[new_column_order].to_csv(utils.get_results_file_name(config), index=False)

if __name__ == '__main__':
    # Load the arguments
    args = parse_args()
    # args = FakeArgs()

    # Instantiate the config object
    config = Config(
        args.dataset, args.model, args.bpr_batch, args.recdim, args.layer, args.dropout, args.keepprob, args.a_fold,
        args.testbatch, args.multicore, args.lr, args.decay, args.pretrain, args.seed, args.epochs, args.load,
        args.checkpoint_path, args.results_path, args.topks, args.tensorboard, args.comment, args.sampling
    )

    # Run the training function
    run_training(config)
