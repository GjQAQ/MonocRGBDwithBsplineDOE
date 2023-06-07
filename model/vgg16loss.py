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
        self.__vgg_blocks = nn.ModuleList([
            vgg16.features[:4].eval(),
            vgg16.features[4:9].eval(),
            vgg16.features[9:16].eval(),
        ])
        self.__vgg_blocks.requires_grad_(False)

        self.__weight = [11.17 / 35.04 / 4, 35.04 / 35.04 / 4, 29.09 / 35.04 / 4]
        self.__loss = None
        # self.diff = None
        # self.total = None
        # self.mean = None
        # self.std = None

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))

        self.add_state('diff', default=torch.tensor([0., 0., 0., 0.]), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor([0, 0, 0., 0.]), dist_reduce_fx='sum')

    def train_loss(self, input_, target):
        self.__common_action(
            input_, target,
            self.__loss_init,
            self.__loss_loop
        )
        return self.__loss

    def update(self, input_: torch.Tensor, target: torch.Tensor):
        self.__common_action(
            input_, target,
            self.__update_init,
            self.__update_loop
        )

    def compute(self) -> torch.Tensor:
        return sum(map(lambda i: self.diff[i] / self.total[i], range(4)))

    def __common_action(self, input_, target, init_callback, loop_callback):
        input_ = (input_ - self.mean) / self.std
        target = (target - self.mean) / self.std
        init_callback(input_, target)

        input_ = functional.pad(input_, (4, 4, 4, 4), 'reflect')
        target = functional.pad(target, (4, 4, 4, 4), 'reflect')
        for i, block in enumerate(self.__vgg_blocks):
            input_, target = block(input_), block(target)
            loop_callback(input_, target, i)

    def __loss_init(self, input_, target):
        self.__loss = functional.l1_loss(input_, target) / 4

    def __loss_loop(self, input_, target, i):
        self.__loss += self.__weight[i] * functional.l1_loss(input_[..., 4:-4, 4:-4], target[..., 4:-4, 4:-4])

    def __update_init(self, input_, target):
        self.diff[0] += (input_ - target).sum() / 4
        self.total[0] += input_.numel()

    def __update_loop(self, input_, target, i):
        self.diff[i + 1] += self.__weight[i] * (input_[..., 4:-4, 4:-4] - target[..., 4:-4, 4:-4]).sum()
        self.total[i + 1] += input_[..., 4:-4, 4:-4].numel()
