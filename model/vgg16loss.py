import torch
import torch.nn as nn
import torch.nn.functional as functional
import torchvision
import pytorch_lightning.metrics as metrics

torch.hub.set_dir('model')


class Vgg16PerceptualLoss(metrics.Metric):
    def __init__(self):
        super().__init__()
        vgg16 = torchvision.models.vgg16(pretrained=True)
        self.vgg_blocks = nn.ModuleList([
            nn.Identity(),
            vgg16.features[:4].eval(),
            vgg16.features[4:9].eval(),
            vgg16.features[9:16].eval(),
        ])
        self.vgg_blocks.requires_grad_(False)

        self.weight = torch.tensor([35.04, 11.17, 35.04, 29.09]) / 35.04 / 4

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1), persistent=False)
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1), persistent=False)

        self.add_state('total', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('loss', default=torch.tensor(0.), dist_reduce_fx='sum')

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        if self.training:
            self.reset()

        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        pred = functional.pad(pred, (4, 4, 4, 4), 'reflect')
        target = functional.pad(target, (4, 4, 4, 4), 'reflect')
        for i, block in enumerate(self.vgg_blocks):
            pred, target = block(pred), block(target)
            self.loss += self.weight[i] * functional.l1_loss(pred[..., 4:-4, 4:-4], target[..., 4:-4, 4:-4])
        self.total += 1

    def compute(self) -> torch.Tensor:
        return self.loss / self.total
