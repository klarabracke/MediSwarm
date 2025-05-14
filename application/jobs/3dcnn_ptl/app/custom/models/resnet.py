from models.base_model import BasicClassifier
import monai.networks.nets as nets
import torch


class ResNet(BasicClassifier):
    """
    ResNet model for classification tasks that uses BasicClassifier's calibrated loss.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        spatial_dims (int): Number of spatial dimensions. Default: 3.
        block (str): ResNet block type. Default: 'basic'.
        layers (list): Number of layers per ResNet block. Default: [3, 4, 6, 3].
        block_inplanes (list): Channel dimensions per block. Default: [64, 128, 256, 512].
        feed_forward (bool): Use feed_forward module in MONAI ResNet. Default: True.
        optimizer: Optimizer class. Default: torch.optim.AdamW.
        optimizer_kwargs (dict): Args for optimizer. Default lr=1e-4.
        lr_scheduler: Scheduler class or None.
        lr_scheduler_kwargs (dict): Args for scheduler.
        aucroc_kwargs (dict): Args for AUROC metric.
        acc_kwargs (dict): Args for Accuracy metric.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        spatial_dims: int = 3,
        block: str = 'basic',
        layers: list = [3, 4, 6, 3],
        block_inplanes: list = [64, 128, 256, 512],
        feed_forward: bool = True,
        optimizer=torch.optim.AdamW,
        optimizer_kwargs: dict = {'lr': 1e-4},
        lr_scheduler=None,
        lr_scheduler_kwargs: dict = {},
        aucroc_kwargs: dict = {"task": "binary"},
        acc_kwargs: dict = {"task": "binary"}
    ):
        
        super().__init__(
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            aucroc_kwargs=aucroc_kwargs,
            acc_kwargs=acc_kwargs,
        )

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.spatial_dims = spatial_dims

        
        self.model = nets.ResNet(
            block=block,
            layers=layers,
            block_inplanes=block_inplanes,
            spatial_dims=spatial_dims,
            in_channels=in_ch,
            conv1_t_size=7,
            conv1_t_stride=1,
            no_max_pool=False,
            norm='BATCH',
            dropout=1.0,
            num_classes=out_ch,
            feed_forward=feed_forward,
            use_conv_h=False,
        )

    def forward(self, x_in: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(x_in)
    