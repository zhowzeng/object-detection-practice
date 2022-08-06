import logging
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from bs4 import BeautifulSoup
from PIL import Image
from torch import nn
from torchvision.models.detection import faster_rcnn, fasterrcnn_resnet50_fpn

logging.basicConfig(level=logging.INFO)


def generate_box(obj):

    xmin = int(obj.find('xmin').text)
    ymin = int(obj.find('ymin').text)
    xmax = int(obj.find('xmax').text)
    ymax = int(obj.find('ymax').text)

    return [xmin, ymin, xmax, ymax]


def generate_label(obj):
    if obj.find('name').text == "with_mask":
        return 1
    elif obj.find('name').text == "mask_weared_incorrect":
        return 2
    return 0


def generate_target(image_id: int, fpath: Path):
    with fpath.open('r') as f:
        data = f.read()
        soup = BeautifulSoup(data, 'xml')
        objects = soup.find_all('object')

        boxes = []
        labels = []
        for i in objects:
            boxes.append(generate_box(i))
            labels.append(generate_label(i))
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        img_id = torch.tensor([image_id])

        # Annotation is in dictionary format
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = img_id

        return target


class FaceMaskDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = Path(root)
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = sorted((self.root / 'images').glob('*.png'))
        self.anns = sorted((self.root / 'annotations').glob('*.xml'))

    def __getitem__(self, idx):
        """load images and targets"""
        img = Image.open(self.imgs[idx]).convert("RGB")
        target = generate_target(image_id=idx, fpath=self.anns[idx])

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)


def collate_fn(batch):
    return tuple(zip(*batch))


def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


class LitFasterRCNN(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        images, targets = batch
        images = list(image for image in images)
        targets = [{k: v for k, v in t.items()} for t in targets]
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        return losses

    def validation_step(self, batch, batch_idx):
        self.model.train()  # in order to return losses rather than detections
        images, targets = batch
        images = list(image for image in images)
        targets = [{k: v for k, v in t.items()} for t in targets]
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        self.log("val_loss", losses, batch_size=len(images))

    def test_step(self, batch, batch_idx):
        images, targets = batch
        images = list(image for image in images)
        targets = [{k: v for k, v in t.items()} for t in targets]
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        self.log("test_loss", losses, batch_size=len(images))

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
        return optimizer


def generate_faster_rcnn_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights=faster_rcnn.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model


train_dataset = FaceMaskDataset('./data/face-mask/', get_transform(train=True))
val_dataset = FaceMaskDataset('./data/face-mask/', get_transform(train=True))
test_dataset = FaceMaskDataset('./data/face-mask/', get_transform(train=False))

indices = torch.randperm(len(train_dataset)).tolist()
train_dataset = torch.utils.data.Subset(train_dataset, indices[:100])
val_dataset = torch.utils.data.Subset(val_dataset, indices[-100:-50])
test_dataset = torch.utils.data.Subset(test_dataset, indices[-50:])

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=collate_fn
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=collate_fn
)

# model
model = LitFasterRCNN(generate_faster_rcnn_model(num_classes=3))

# train model
trainer = pl.Trainer(accelerator='cpu', max_epochs=1)
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

# test model
trainer.test(dataloaders=test_loader)
