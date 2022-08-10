import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchvision.models.detection import faster_rcnn, fasterrcnn_resnet50_fpn


class LitFasterRCNN(pl.LightningModule):
    def __init__(self, faster_rcnn):
        super().__init__()
        self.save_hyperparameters(ignore=['faster_rcnn'])
        self.model = faster_rcnn

    def training_step(self, batch, batch_idx):
        images, targets = batch
        images = list(image for image in images)
        targets = [{k: v for k, v in t.items()} for t in targets]
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        self.log("train_loss", losses, batch_size=len(images))
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
        self.model.train()  # in order to return losses rather than detections
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
