# Based on this notebook:
# https://www.kaggle.com/code/shoheiazuma/stable-diffusion-vit-baseline-train

import os
import gc
import shutil
import torch
import hydra
import mlflow
import numpy as np
import pandas as pd
from torch import nn
from PIL import Image
from tqdm import tqdm
from scipy import spatial
from torchvision import transforms
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from omegaconf import DictConfig, OmegaConf, open_dict
from helpers.log import log_params_to_mlflow


import timm
from timm.utils import AverageMeter

import warnings
warnings.filterwarnings('ignore')


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class CustomDataset(Dataset):
    def __init__(self, image_list, prompt_list, processor):
        self.image_list = image_list
        self.prompt_list = prompt_list
        self.processor = processor

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        image = Image.open(image_path)
        image = self.processor(image)
        prompt = self.prompt_list[index]
        return image, prompt


class Collator:
    def __init__(self):
        self.st_model = SentenceTransformer(
            'weights/sentence-transformers-222/all-MiniLM-L6-v2',
            device='cpu'
        )

    def __call__(self, batch):
        images, prompts = zip(*batch)
        images = torch.stack(images)
        prompt_embeddings = self.st_model.encode(
            prompts,
            show_progress_bar=False,
            convert_to_tensor=True
        )
        return images, prompt_embeddings


def cosine_similarity(y_trues, y_preds):
    return np.mean([
        1 - spatial.distance.cosine(y_true, y_pred)
        for y_true, y_pred in zip(y_trues, y_preds)
    ])


def get_dataloaders(
    val_image_list,
    val_prompt_list,
    val_batch_size,
    train_image_list,
    train_prompt_list,
    train_batch_size,
    n_workers,
):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    trn_dataset = CustomDataset(train_image_list, train_prompt_list, transform)
    val_dataset = CustomDataset(val_image_list, val_prompt_list, transform)
    collator = Collator()

    dataloaders = {}
    dataloaders['train'] = DataLoader(
        dataset=trn_dataset,
        shuffle=True,
        batch_size=train_batch_size,
        pin_memory=True,
        num_workers=n_workers,
        drop_last=True,
        collate_fn=collator
    )
    dataloaders['val'] = DataLoader(
        dataset=val_dataset,
        shuffle=False,
        batch_size=val_batch_size,
        pin_memory=True,
        num_workers=n_workers,
        drop_last=False,
        collate_fn=collator
    )

    return dataloaders


def train_one_epoch(
        model,
        optimizer,
        scheduler,
        dataloader,
        device,
        epoch,
        n_accumulate=1,
        epoch_length=1000
    ):
    model.train()
    dataset_size = 0
    running_loss = 0.0
    bar = tqdm(range(epoch_length), total=epoch_length)
    for step in bar:
        data = next(iter(dataloader))
        input_ids = data['input_ids'].to(device)
        pixel_values = data['pixel_values'].to(device)
        batch_size = input_ids.size(0)
        outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=input_ids)
        loss = outputs.loss
        loss = loss / n_accumulate
        loss.backward()
        if (step + 1) % n_accumulate == 0:
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None: scheduler.step()
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        epoch_loss = running_loss / dataset_size
        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss, LR=optimizer.param_groups[0]['lr'])
    gc.collect()
    return epoch_loss



def train(
    model,
    optimizer,
    scheduler,
    num_epochs,
    epoch_length,
    train_loader,
    valid_loader,
    device,
    patience=3,
    checkpoints_dir_path="checkpoints",
    max_saved_models=5
):
    best_score = -1.0
    prev_score = -1.0
    criterion = nn.CosineEmbeddingLoss()

    for epoch in range(num_epochs):

        train_meters = {
            'loss': AverageMeter(),
            'cos': AverageMeter(),
        }
        model.train()

        bar = tqdm(range(epoch_length), total=epoch_length)
        for step in bar:
            X, y = next(iter(train_loader))
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            X_out = model(X)
            target = torch.ones(X.size(0)).to(device)
            loss = criterion(X_out, y, target)
            loss.backward()

            optimizer.step()
            scheduler.step()

            trn_loss = loss.item()
            trn_cos = cosine_similarity(
                X_out.detach().cpu().numpy(),
                y.detach().cpu().numpy()
            )

            train_meters['loss'].update(trn_loss, n=X.size(0))
            train_meters['cos'].update(trn_cos, n=X.size(0))

            bar.set_postfix(
                Epoch=epoch,
                Train_Loss=train_meters['loss'].avg,
                Train_Cos=train_meters['cos'].avg,
                LR=optimizer.param_groups[0]['lr']
            )

        mlflow.log_metric("train_loss", train_meters['loss'].avg, step=epoch)
        mlflow.log_metric("train_cos", train_meters['cos'].avg, step=epoch)

        val_meters = {
            'loss': AverageMeter(),
            'cos': AverageMeter(),
        }

        model.eval()
        bar = tqdm(valid_loader, total=len(valid_loader))
        for X, y in bar:
            X, y = X.to(device), y.to(device)

            with torch.no_grad():
                X_out = model(X)
                target = torch.ones(X.size(0)).to(device)
                loss = criterion(X_out, y, target)

                val_loss = loss.item()
                val_cos = cosine_similarity(
                    X_out.detach().cpu().numpy(),
                    y.detach().cpu().numpy()
                )

            val_meters['loss'].update(val_loss, n=X.size(0))
            val_meters['cos'].update(val_cos, n=X.size(0))
            bar.set_postfix(
                Epoch=epoch,
                Val_Loss=val_meters['loss'].avg,
                Val_Cos=val_meters['cos'].avg,
            )

        mlflow.log_metric("val_loss", val_meters['loss'].avg, step=epoch)
        mlflow.log_metric("val_cos", val_meters['cos'].avg, step=epoch)

        if val_meters['cos'].avg > best_score:
            print(f"Validation Coss Improved ({best_score:.4f} ---> {val_meters['cos'].avg:.4f})")
            best_score = val_meters['cos'].avg

            model_path = os.path.join(checkpoints_dir_path, f"model_{epoch}")
            print(f"Saving model to {model_path}")
            mlflow.pytorch.save_model(model, model_path)

            saved_models = sorted(os.listdir(checkpoints_dir_path), key=lambda x: int(x.split("_")[-1]))
            while len(saved_models) > max_saved_models:
                model_to_remove = saved_models.pop(0)
                shutil.rmtree(os.path.join(checkpoints_dir_path, model_to_remove))

            mlflow.log_metric("val_cos_max", best_score, step=epoch)

        if val_meters['cos'].avg > prev_score:
            prev_score = val_meters['cos'].avg
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping after {patience} epochs without improvement.")
            break

    print(f'Best cos: {best_score:.4f}, best loss: {val_meters["loss"].avg:.4f}')


CONFIG_DIR = os.getenv('CONFIG_DIR')
CONFIG_NAME = os.getenv('CONFIG_NAME')

assert CONFIG_DIR is not None and CONFIG_DIR != '', 'CONFIG_DIR is not set'
assert CONFIG_NAME is not None and CONFIG_NAME != '', 'CONFIG_NAME is not set'

@hydra.main(config_path=CONFIG_DIR, config_name=CONFIG_NAME, version_base=None)
def main(args : DictConfig):
    print(OmegaConf.to_yaml(args))

    mlflow.set_tracking_uri(args.tracking_uri)
    if args.use_mlflow:
        experiment = mlflow.set_experiment(args.experiment_name)
        mlflow_run = mlflow.start_run(experiment_id=experiment.experiment_id)
        mlflow_run_id = mlflow_run.info.run_id
        with open_dict(args):
            args.run_id = mlflow_run_id
        log_params_to_mlflow(OmegaConf.to_container(args))
        checkpoints_dir_path = mlflow_run.info.artifact_uri

        if args.is_cluster:
            # костыль, потому что тренька на кластере, а на млфлоу на локалке и до маунта разные пути блять...
            checkpoints_dir_path = checkpoints_dir_path.replace('/mnt/home', '/mnt/home/divashkov')
    else:
        checkpoints_dir_path = os.path.join(args.checkpoints_dir, 'tmp')

    # Set seed
    set_seed(args.seed)

    # Read index file
    train_index = pd.read_csv(args.train_index_path)
    val_index = pd.read_csv(args.eval_index_path)

    # Get rid of images with no prompt
    train_index = train_index[~train_index['Prompt'].isna()]
    val_index = val_index[~val_index['Prompt'].isna()]

    val_image_list = [os.path.join(args.data_path, 'eval_images', x) for x in val_index['image_name']]
    val_prompt_list = list(val_index['Prompt'])

    train_image_list = [os.path.join(args.data_path, 'train_images', x) for x in train_index['image_name']]
    train_prompt_list = list(train_index['Prompt'])

    # Set dataloaders
    dataloaders = get_dataloaders(
        val_image_list,
        val_prompt_list,
        args.valid_batch_size,
        train_image_list,
        train_prompt_list,
        args.train_batch_size,
        n_workers=args.n_workers,
    )
    val_dataloader = dataloaders['val']
    train_dataloader = dataloaders['train']

    # Load model
    device = args.device

    model = timm.create_model(
        args.model_name,
        pretrained=True,
        num_classes=384
    )
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    model.set_grad_checkpointing()
    model.to(device)

    # Load optimizer, scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.t_max,
        eta_min=args.min_lr
    )

    train(
        model,
        optimizer,
        scheduler,
        args.epochs,
        args.epoch_length,
        train_dataloader,
        val_dataloader,
        device,
        args.patience,
        checkpoints_dir_path,
        max_saved_models=5
    )

    mlflow.end_run()


if __name__ == "__main__":
    main()