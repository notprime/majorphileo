import argparse
import os
import random
import sys
from datetime import datetime
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist

import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from constants import *
from resources_paths import *
from training_setup import get_models, get_models_pretrained, get_trainer
from utils import data_protocol
from utils import load_data


class PytorchDistributedTrainer:
    def __init__(self):

        self.init_dist()
        self.init_args()
        self.init_seed()
        self.init_model()
        self.init_loader()
        self.init_wandb()

        self.init_trainer()
        self.trainer.train()
        self.trainer.test()
        self.trainer_save_info()

        self.cleanup()

    def init_dist(self):
        if 'LOCAL_RANK' in os.environ:
            self.WORLD_SIZE = int(os.environ['WORLD_SIZE'])
            self.RANK = int(os.environ['RANK'])
            self.LOCAL_RANK = int(os.environ['LOCAL_RANK'])
        elif 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
            self.WORLD_SIZE = int(os.environ['OMPI_COMM_WORLD_SIZE'])
            self.RANK = int(os.environ['OMPI_COMM_WORLD_RANK'])
            self.LOCAL_RANK = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        else:
            sys.exit("Can't find the environment variables!")

        try:
            dist.init_process_group(backend='nccl', rank=self.RANK,
                                    timeout=timedelta(minutes=180),
                                    world_size=self.WORLD_SIZE)
            print(f"[Rank {self.RANK}] init_process_group succeeded")
        except Exception as e:
            import traceback
            print(f"[Rank {self.RANK}] init_process_group failed:\n{traceback.format_exc()}")
        self.DEVICE = 'cuda:{}'.format(self.LOCAL_RANK)
        torch.cuda.set_device(self.LOCAL_RANK)

        self.sync_datetime()

    def sync_datetime(self):

        # Create the datetime string
        current_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        # Convert string to a list of ASCII values
        current_time_tensor = torch.tensor([ord(c) for c in current_time], dtype=torch.int)
        # Print the datetime on each rank
        print(f"Rank {self.RANK}: current datetime is {current_time}")

        current_time_tensor = current_time_tensor.cuda()
        if not current_time_tensor.is_contiguous():
            current_time_tensor = current_time_tensor.contiguous()

        # Broadcast the tensor from rank 0 to all other ranks
        dist.broadcast(current_time_tensor, src=0)

        # Convert back from tensor to string
        current_time = ''.join([chr(c) for c in current_time_tensor.tolist()])
        # Print the datetime on each rank
        print(f"Rank {self.RANK}: datetime after broadcasting from rank 0 is {current_time}")

        # Assign to the object's datetime attribute
        self.datetime = current_time

    def init_args(self):
        # TODO select good default values
        # TODO modify trainer and training loops to incorporate other parameters (commented out)

        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument('--random_seed', type=int, default=123456)  # NEW, from hardcoded
        parser.add_argument('--base_path', type=str, default=BASE_PATH)  # NEW. from hardcoded
        parser.add_argument('--experiment_name', type=str, default=f'{self.datetime}-experiment')
        parser.add_argument('--model_name', type=str, default=MODEL_LIST[0], choices=MODEL_LIST)  # new DEFAULT, from required=True
        parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to pretrained weights')
        #parser.add_argument('--freeze_pretrained', type=bool, default=False, help='freeze pretrained model weights') # action="store_true", help='freeze pretrained model weights')
        parser.add_argument('--freeze_pretrained', action="store_true", help='freeze pretrained model weights')

        parser.add_argument('--epochs', type=int, default=250)
        parser.add_argument('--batch_size', type=int, default=16, help="batch size per GPU")
        parser.add_argument('--learning_rate', type=float, default=1e-3)  # RENAMED FROM 'lr'
        # parser.add_argument('--learning_rate_end', type=float, default=1e-6)  # NEW
        parser.add_argument('--lr_scheduler', type=str, default=None, choices=[None] + LR_SCHEDULERS)
        # parser.add_argument('--weight_decay', type=float, default=1e-2)   # NEW
        parser.add_argument('--warmup_epochs', type=int, default=5)  # RENAMED from 'warmup_steps'
        # parser.add_argument('--warmup_lr_start', type=float, default=1e-6)    # NEW
        parser.add_argument('--warmup_gamma', type=int, default=10)
        parser.add_argument('--es_patience', type=int, default=50)  # RENAMED from 'early_stop'
        parser.add_argument('--es_delta', type=float, default=0.0)  # NEW
        # REMOVED 'warmup'

        parser.add_argument('--data_path', type=str, default=PATH_DS_DOWNSTREAM_NP_128_10M, choices=PATHS_DS_DOWNSTREAM)  # NEW, aggregates various data_paths_XXX_XXX
        parser.add_argument('--img_size', type=int, default=128)  # RENAMED from 'input_size'
        parser.add_argument('--input_channels', type=int, default=10, help='Number of input channels')
        parser.add_argument('--augmentations', action="store_true", help='enables augmentations')
        parser.add_argument('--num_workers', type=int, default=0)
        # REMOVED 'output_channels' (depends on downstream task)

        parser.add_argument('--downstream_task', type=str, default=DOWNSTREAM_LIST[0], choices=DOWNSTREAM_LIST)  # new DEFAULT from required=True
        parser.add_argument('--regions', nargs='+', default=None, choices=REGIONS, help='select regions to be included')
        parser.add_argument('--n_shot', type=int, default=None, help='Loads n-samples of data from the specified geographic region(s)')
        parser.add_argument('--split_ratio', type=float, default=None, help='Loads a percentage of the data from specified geographic region(s)')

        # REMOVED 'model_device'
        # REMOVED 'generator_device'
        # REMOVED 'C'
        # REMOVED 'data_parallel'
        # REMOVED 'device_ids'

        parser.add_argument('--vis_val', action="store_true", help='enable saving of intermediate visualization plots')

        parser.add_argument('--wandb_project', type=str, default='PhilEO-Eval')  # NEW

        self.params = parser.parse_args()
        self.check_args()
        self.preprocess_args()

    def check_args(self) -> None:

        # -------- parameter-specific checks --------

        assert self.params.random_seed >= 0, f"Argument 'random_seed' must be a non-negative integer. Got {self.params.random_seed}"
        assert os.path.exists(self.params.base_path), f"Argument 'base_path' must be a valid path. Got {self.params.base_path}"
        assert self.params.pretrained_model_path is None or os.path.exists(self.params.pretrained_model_path), f"Argument 'pretrained_model_path' must be None or a valid path. Got {self.params.pretrained_model_path}"

        assert self.params.epochs > 0, f"Argument 'epochs' must be must be a positive integer. Got {self.params.epochs}"
        assert self.params.batch_size > 0, f"Argument 'batch_size must be a positive integer. Got {self.params.batch_size}"
        assert self.params.learning_rate > 0, f"Argument 'learning_rate' must be a positive value. Got {self.params.learning_rate}"
        # assert self.params.learning_rate_end > 0, f"Argument 'learning_rate_end' must be a positive value. Got {self.params.learning_rate_end}"
        # assert self.params.weight_decay > 0, f"Argument 'weight_decay' must be a positive value. Got {self.params.weight_decay}"
        assert self.params.warmup_epochs >= 0, f"Argument 'warmup_epochs' must be a non-negative integer. Got {self.params.warmup_epochs}"
        # assert self.params.warmup_lr_start > 0, f"Argument 'warmup_lr_start' must be a positive value. Got {self.params.warmup_lr_start}"
        assert self.params.warmup_gamma > 0, f"Argument 'warmup_gamma' must be must be a positive integer. Got {self.params.warmup_gamma}"
        assert self.params.es_patience > 0, f"Argument 'es_patience' must be must be a positive integer. Got {self.params.es_patience}"
        assert self.params.es_delta >= 0, f"Argument 'es_delta' must be a non-negative value. Got {self.params.es_delta}"

        assert os.path.exists(self.params.data_path), f"Argument 'data_path' must be a valid path. Got {self.params.data_path}"
        assert self.params.img_size >= 0, f"Argument 'img_size' must be a non-negative integer. Got {self.params.img_size}"
        assert self.params.input_channels > 0, f"Argument 'input_channels' must be must be a positive integer. Got {self.params.input_channels}"
        assert self.params.num_workers >= 0, f"Argument 'num_workers' must be a non-negative integer. Got {self.params.num_workers}"

        assert self.params.n_shot is None or self.params.n_shot > 0, f"Argument 'n_shot' must be None or a positive integer. Got {self.params.n_shot}"
        assert self.params.split_ratio is None or 0.0 < self.params.split_ratio <= 1.0, f"Argument 'split_ratio' must be None or a value in (0.0, 1.0]. Got {self.params.split_ratio}"

        assert self.params.regions is None or set(self.params.regions).issubset(set(REGIONS)), f"Argument 'regions' must be None or a subset of {REGIONS}. Got {self.params.regions}"

        # -------- intra-parameters checks --------

        assert (self.params.n_shot is not None) ^ (self.params.split_ratio is not None), \
            f"Must set either n_shot or split_ratio (not both, not neither). Got {self.params.n_shot} and {self.params.split_ratio}"

        if self.params.pretrained_model_path is not None:
            assert self.params.model_name in MODEL_LIST_PRETRAINED, \
                f"When specifying a 'pretrained_model_path', the 'model_name' must be one of a pretrained model from {MODEL_LIST_PRETRAINED}"

    def preprocess_args(self) -> None:

        if self.RANK == 0:
            print("Original Arguments:")
            print(self.params)

        #######

        self.params.init_lr = self.params.learning_rate     # original value, for logging purposes in save_info()

        self.params.learning_rate = self.params.learning_rate * np.sqrt(self.WORLD_SIZE)    # TODO <<<<<<<<
        # self.params.learning_rate_end = self.params.learning_rate_end * np.sqrt(self.WORLD_SIZE)

        # overwrite output_channels depending on the downstream task
        if self.params.downstream_task == 'lc':
            self.params.output_channels = 11    # overwrite arg
        elif self.params.downstream_task == 'roads' or self.params.downstream_task == 'building':
            self.params.output_channels = 1     # overwrite arg
        else:
            raise ValueError(f"Error in setting number of output channels based on undefined task {self.params.downstream_task}")

        print(f"Set 'output_channels' to {self.params.output_channels} for task {self.params.downstream_task.upper()}")

        if self.params.warmup_epochs > 0:
            self.params.learning_rate = self.params.learning_rate / int(self.params.warmup_gamma ** self.params.warmup_epochs)  # for warmup start

        #######

        if self.RANK == 0:
            print("Preprocessed Arguments:")
            print(self.params)

    def init_seed(self):
        np.random.seed(self.params.random_seed)
        os.environ['PYTHONHASHSEED'] = str(self.params.random_seed)
        random.seed(self.params.random_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(self.params.random_seed)
        torch.manual_seed(self.params.random_seed)

    def init_model(self):

        if self.params.pretrained_model_path is not None:
            print("Loading pre-trained model... ")
            print(f"Debugging - value for freeze_pretrained is: {self.params.freeze_pretrained}")
            model, summary = get_models_pretrained(
                model_name=self.params.model_name,
                input_channels=self.params.input_channels,
                output_channels=self.params.output_channels,
                input_size=self.params.img_size,
                path_model_weights=self.params.pretrained_model_path,
                freeze=self.params.freeze_pretrained
            )

            if self.params.model_name == 'GeoAware_contrastive_core_nano' or self.params.model_name == 'GeoAware_contrastive_core_nano_classifier':
                self.NAME = model.__class__.__name__ + '_contrastive_frozen' if self.params.freeze_pretrained else model.__class__.__name__ + '_contrastive_unfrozen'
            elif self.params.model_name == 'GeoAware_mh_pred_core_nano' or self.params.model_name == 'GeoAware_mh_pred_core_nano_classifier':
                self.NAME = model.__class__.__name__ + '_mh_pred_frozen' if self.params.freeze_pretrained else model.__class__.__name__ + '_mh_pred_unfrozen'
            else:
                self.NAME = model.__class__.__name__ + '_frozen' if self.params.freeze_pretrained else model.__class__.__name__ + '_unfrozen'

        else:
            print(f"No pretrained model was supplied -> Ignoring 'freeze_pretrained' argument")
            model, summary = get_models(
                model_name=self.params.model_name,
                input_channels=self.params.input_channels,
                output_channels=self.params.output_channels,
                input_size=self.params.img_size,
            )

            self.NAME = model.__class__.__name__

        # Convert batchnorm to syncbatchnorm
        # https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)  # , process_group)
        self.model = DDP(model.to(self.DEVICE), device_ids=[self.LOCAL_RANK], output_device=self.LOCAL_RANK, find_unused_parameters=True)

        self.model_summary = summary

        self.output_folder = f'{self.params.base_path}/{self.params.experiment_name}/{self.params.downstream_task}/{self.datetime}_{self.NAME}_{self.params.downstream_task}'
        if self.params.lr_scheduler is not None:
            self.output_folder = self.output_folder + f"_{self.params.lr_scheduler}"

    def init_loader(self):
        #TODO IMPROVE: find a way to skip resaving of dataset for sharing across DDP

        # NB: dataset NOT assigned automatically depending on 'model_name' to allow
        # the same model to be both trained and evaluated at different image sizes
        dataset_folder = self.params.data_path
        dataset_name = DATASET_DS_DOWNSTREAM_NAMES[self.params.data_path]

        # ------------- update output folder depending on downstream task ---------------

        if self.params.n_shot is not None:
            self.output_folder = f'{self.output_folder}_{self.params.n_shot}'
        elif self.params.split_ratio is not None:
            self.output_folder = f'{self.output_folder}_{self.params.split_ratio}'

        # ------------- process at rank 0 creates the dataset depending on downstream task, saves to shared location ---

        if DATASETS_PATH:
            # Datasets already created, just load them
            if self.RANK == 0:
                print("Loading datasets already created... ")
            path = os.path.join(DATASETS_PATH,                       # base path
                                self.params.downstream_task,         # downstream task
                                f"dataset_{self.params.n_shot}")    # n_shot
            x_train = torch.load(os.path.join(path, "datasets_x_train.pt"), weights_only=False)
            y_train = torch.load(os.path.join(path, "datasets_y_train.pt"), weights_only=False)
            x_val = torch.load(os.path.join(path, "datasets_x_val.pt"), weights_only=False)
            y_val = torch.load(os.path.join(path, "datasets_y_val.pt"), weights_only=False)
            x_test = torch.load(os.path.join(path, "datasets_x_test.pt"), weights_only=False)
            y_test = torch.load(os.path.join(path, "datasets_y_test.pt"), weights_only=False)
            print(f"RANK {self.RANK}: loaded the dataset")

        else:
            # Datasets need to be created and loaded
            tmp_dataset_folder = os.path.join(self.output_folder, "datasets_")
            tmp_dataset_ext = ".pt"
            self.tmp_dataset_folder = tmp_dataset_folder
            self.tmp_dataset_ext = tmp_dataset_ext

            if self.RANK == 0:
                #"""
                print("Creating datasets from scratch... ")
                if self.params.n_shot is not None:
                    x_train, y_train, x_val, y_val = data_protocol.protocol_fewshot_memmapped(
                        folder=dataset_folder,
                        dst=PATH_DS_DOWNSTREAM_NSHOT,
                        n=self.params.n_shot,
                        regions=self.params.regions,
                        y=self.params.downstream_task,
                        data_selection='create',
                        name=dataset_name
                    )
                elif self.params.split_ratio is not None:
                    x_train, y_train, x_val, y_val = data_protocol.protocol_split(
                        dataset_folder,
                        split_percentage=self.params.split_ratio,
                        regions=self.params.regions,
                        y=self.params.downstream_task
                    )

                x_test, y_test = data_protocol.get_testset(
                    folder=dataset_folder,
                    y=self.params.downstream_task
                )

                os.makedirs(self.output_folder, exist_ok=False)  # should not have been created yet, depends on exp datetime
                # Save the datasets to a shared location, so that all processes can see the data
                torch.save(x_train, tmp_dataset_folder + "x_train" + tmp_dataset_ext)
                torch.save(y_train, tmp_dataset_folder + "y_train" + tmp_dataset_ext)
                torch.save(x_val, tmp_dataset_folder + "x_val" + tmp_dataset_ext)
                torch.save(y_val, tmp_dataset_folder + "y_val" + tmp_dataset_ext)
                torch.save(x_test, tmp_dataset_folder + "x_test" + tmp_dataset_ext)
                torch.save(y_test, tmp_dataset_folder + "y_test" + tmp_dataset_ext)
                print(f"Rank {self.RANK}: Created x_train, y_train, x_val, y_val, x_test, y_test")

            # -------------
            # SYNC BARRIER, datasets created.
            # each process loads the shared dataset.
            # Rank 0 deletes the tmp save file used for sharing the dataset
            # -------------

            # synchronization point, the datasets must now have been created and saved by process 0 before continuing
            dist.barrier()
            x_train = torch.load(os.path.join(self.output_folder, "datasets_x_train.pt"), weights_only=False)
            y_train = torch.load(os.path.join(self.output_folder, "datasets_y_train.pt"), weights_only=False)
            x_val = torch.load(os.path.join(self.output_folder, "datasets_x_val.pt"), weights_only=False)
            y_val = torch.load(os.path.join(self.output_folder, "datasets_y_val.pt"), weights_only=False)
            x_test = torch.load(os.path.join(self.output_folder, "datasets_x_test.pt"), weights_only=False)
            y_test = torch.load(os.path.join(self.output_folder, "datasets_y_test.pt"), weights_only=False)
            print(f"RANK {self.RANK}: loaded the dataset")

        # ------------- datasets creation ---

        train_dataset, test_dataset, val_dataset = load_data.load_datasets(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            with_augmentations=self.params.augmentations,
            downstream_task=self.params.downstream_task,
            model_name=self.params.model_name.split('_')[0],
        )
        print(f"RANK {self.RANK}: Created datasets")

        # ------------- train dataloader ---------------

        train_sampler = DistributedSampler(
            dataset=train_dataset,
            rank=self.RANK,
            num_replicas=self.WORLD_SIZE,
            shuffle=True,
        )

        self.train_dataloader = DataLoader(
            dataset=train_dataset,
            sampler=train_sampler,
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers,
            pin_memory=True,
            # generator=torch.Generator(device=self.DEVICE),
        )

        print(f"RANK {self.RANK}: Created train dataloader")
        print(f"RANK {self.RANK}: len train dataloader {len(self.train_dataloader)}")

        # ------------- test dataloader ---------------

        test_sampler = DistributedSampler(
            dataset=test_dataset,
            rank=self.RANK,
            num_replicas=self.WORLD_SIZE,
            shuffle=False
        )

        self.test_dataloader = DataLoader(
            dataset=test_dataset,
            sampler=test_sampler,
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers,
            pin_memory=True,
        )

        print(f"RANK {self.RANK}: Created test dataloader")
        print(f"RANK {self.RANK}: len test dataloader {len(self.test_dataloader)}")


        # ------------- val dataloader ---------------

        val_sampler = DistributedSampler(
            dataset=val_dataset,
            rank=self.RANK,
            num_replicas=self.WORLD_SIZE,
            shuffle=False,
        )

        self.val_dataloader = DataLoader(
            dataset=val_dataset,
            sampler=val_sampler,
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers,
            pin_memory=True,
        )

        print(f"RANK {self.RANK}: Created val dataloader")
        print(f"RANK {self.RANK}: len val dataloader {len(self.val_dataloader)}")
        
    def init_wandb(self):
        self.wandb=None     # set default for all processes
        if self.RANK == 0:
            os.environ["WANDB_SILENT"] = "true"
            wandb.require("service")
            self.wandb = wandb.init(
                project=self.params.wandb_project,
                name=self.output_folder,
                config=vars(self.params),
            )
            self.wandb.watch(self.model, log="all")

    def init_trainer(self):
        self.trainer = get_trainer(
            model_name=self.params.model_name,
            downstream_task=self.params.downstream_task,
            epochs=self.params.epochs,
            lr=self.params.learning_rate,
            model=self.model,
            device=self.DEVICE,
            lr_scheduler=self.params.lr_scheduler,
            # removed 'warmup'
            es_patience=self.params.es_patience,    # renamed from 'early_stop'
            es_delta=self.params.es_delta,  # new
            dl_train=self.train_dataloader,
            dl_val=self.val_dataloader,
            dl_test=self.test_dataloader,
            NAME=self.NAME,
            output_folder=self.output_folder,
            vis_val=self.params.vis_val,
            warmup_epochs=self.params.warmup_epochs,    # renamed from 'warmup_steps'
            warmup_gamma=self.params.warmup_gamma,
            RANK=self.RANK,     # new
            wandb=self.wandb,    # new
        )

    def trainer_save_info(self):
        self.trainer.save_info(
            model_summary=self.model_summary,
            n_shot=self.params.n_shot,
            p_split=self.params.split_ratio,
            warmup_epochs=self.params.warmup_epochs > 0,
            lr=self.params.init_lr
        )

    def cleanup(self):
        dist.barrier()
        dist.destroy_process_group()
        if DATASETS_PATH is None and self.RANK == 0:
            os.remove(self.tmp_dataset_folder + "x_train" + self.tmp_dataset_ext)
            os.remove(self.tmp_dataset_folder + "y_train" + self.tmp_dataset_ext)
            os.remove(self.tmp_dataset_folder + "x_val" + self.tmp_dataset_ext)
            os.remove(self.tmp_dataset_folder + "y_val" + self.tmp_dataset_ext)
            os.remove(self.tmp_dataset_folder + "x_test" + self.tmp_dataset_ext)
            os.remove(self.tmp_dataset_folder + "y_test" + self.tmp_dataset_ext)
            self.wandb.finish()


if __name__ == "__main__":
    PytorchDistributedTrainer()
