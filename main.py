import logging
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch

from src.data import FaceMaskDataset, collate_fn, get_transform
from src.model import LitFasterRCNN, generate_faster_rcnn_model

logging.basicConfig(level=logging.INFO)


if __name__ == '__main__':
    parser = ArgumentParser()

    # add PROGRAM level args
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)

    # add model specific args
    parser = LitFasterRCNN.add_model_specific_args(parser)

    # add all the available trainer options to argparse
    # ie: now --accelerator --devices --num_nodes ... --fast_dev_run all work in the cli
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    train_dataset = FaceMaskDataset('./data/face-mask/', get_transform(train=True))
    val_dataset = FaceMaskDataset('./data/face-mask/', get_transform(train=True))
    test_dataset = FaceMaskDataset('./data/face-mask/', get_transform(train=False))

    indices = torch.randperm(len(train_dataset)).tolist()
    train_dataset = torch.utils.data.Subset(train_dataset, indices[:-100])
    val_dataset = torch.utils.data.Subset(val_dataset, indices[-100:-50])
    test_dataset = torch.utils.data.Subset(test_dataset, indices[-50:])

    logging.info(f'{len(train_dataset)=}; {len(val_dataset)=}; {len(test_dataset)=}')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    logging.info(f'{len(train_loader)=}; {len(val_loader)=}; {len(test_loader)=}')

    # model
    model = LitFasterRCNN(args, model=generate_faster_rcnn_model(num_classes=3))

    # train model
    trainer = pl.Trainer.from_argparse_args(
        args,
        default_root_dir='./checkpoints/',
        profiler='simple',
        callbacks=[
            pl.callbacks.early_stopping.EarlyStopping(monitor="val_loss", mode="min"),
            pl.callbacks.DeviceStatsMonitor(),
        ],
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # test model
    trainer.test(dataloaders=test_loader)

    # -------------------------------------------------------------------------------------------------

    # # load checkpoint
    # model = MyLightningModule.load_from_checkpoint("/path/to/checkpoint.ckpt")

    # checkpoint = torch.load(CKPT_PATH)
    # print(checkpoint.keys())

    # checkpoint = torch.load(CKPT_PATH)
    # encoder_weights = checkpoint["model"]

    # # automatically restores model, epoch, step, LR schedulers, apex, etc...
    # trainer.fit(model, ckpt_path="some/path/to/my_checkpoint.ckpt")
