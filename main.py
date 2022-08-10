import logging

import pytorch_lightning as pl
import torch

from src.data import FaceMaskDataset, collate_fn, get_transform
from src.model import LitFasterRCNN, generate_faster_rcnn_model

logging.basicConfig(level=logging.INFO)


if __name__ == '__main__':

    train_dataset = FaceMaskDataset('./data/face-mask/', get_transform(train=True))
    val_dataset = FaceMaskDataset('./data/face-mask/', get_transform(train=True))
    test_dataset = FaceMaskDataset('./data/face-mask/', get_transform(train=False))

    indices = torch.randperm(len(train_dataset)).tolist()
    train_dataset = torch.utils.data.Subset(train_dataset, indices[:100])
    val_dataset = torch.utils.data.Subset(val_dataset, indices[-100:-50])
    test_dataset = torch.utils.data.Subset(test_dataset, indices[-50:])

    logging.info(f'{len(train_dataset)=}; {len(val_dataset)=}; {len(test_dataset)=}')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=True, num_workers=8, collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=2, shuffle=False, num_workers=8, collate_fn=collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=2, shuffle=False, num_workers=8, collate_fn=collate_fn
    )

    logging.info(f'{len(train_loader)=}; {len(val_loader)=}; {len(test_loader)=}')

    # model
    model = LitFasterRCNN(generate_faster_rcnn_model(num_classes=3))

    # train model
    trainer = pl.Trainer(
        accelerator='cpu',
        max_epochs=5,
        default_root_dir='./checkpoints/',
        callbacks=[pl.callbacks.early_stopping.EarlyStopping(monitor="val_loss", mode="min")],
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
