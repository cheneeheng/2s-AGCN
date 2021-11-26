import torch


class BatchNorm1d(torch.nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1,
                 weight_freeze=False, bias_freeze=False,
                 weight_init=1.0, bias_init=0.0):
        super().__init__(num_features, eps=eps, momentum=momentum)
        if weight_init is not None:
            self.weight.data.fill_(weight_init)
        if bias_init is not None:
            self.bias.data.fill_(bias_init)
        self.weight.requires_grad = not weight_freeze
        self.bias.requires_grad = not bias_freeze


class GhostBatchNorm1d(BatchNorm1d):
    def __init__(self, num_features, num_splits=16, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        self.register_buffer('running_mean', torch.zeros(
            num_features*self.num_splits))
        self.register_buffer('running_var', torch.ones(
            num_features*self.num_splits))

    # def train(self, mode=True):
    #     # lazily collate stats when we are going to use them
    #     # This calculates the mean + var when the model.eval() is called.
    #     if (self.training is True) and (mode is False):
    #         self.running_mean = torch.mean(
    #             self.running_mean.view(self.num_splits, self.num_features),
    #             dim=0
    #         ).repeat(self.num_splits)
    #         self.running_var = torch.mean(
    #             self.running_var.view(self.num_splits, self.num_features),
    #             dim=0
    #         ).repeat(self.num_splits)
    #     return super().train(mode)

    def forward(self, input):
        N, C, K = input.shape
        if self.training or not self.track_running_stats:
            # return torch.nn.functional.batch_norm(
            #     input.view(-1, C*self.num_splits, K),
            #     running_mean=self.running_mean,
            #     running_var=self.running_var,
            #     weight=self.weight.repeat(self.num_splits),
            #     bias=self.bias.repeat(self.num_splits),
            #     training=True,
            #     momentum=self.momentum,
            #     eps=self.eps).view(N, C, K)
            bn = torch.nn.functional.batch_norm(
                input.view(-1, C*self.num_splits, K),
                running_mean=self.running_mean,
                running_var=self.running_var,
                weight=self.weight.repeat(self.num_splits),
                bias=self.bias.repeat(self.num_splits),
                training=True,
                momentum=self.momentum,
                eps=self.eps)
            self.running_mean = torch.mean(
                self.running_mean.view(self.num_splits, self.num_features),
                dim=0
            ).repeat(self.num_splits)
            self.running_var = torch.mean(
                self.running_var.view(self.num_splits, self.num_features),
                dim=0
            ).repeat(self.num_splits)
            return bn.view(N, C, K)
        else:
            return torch.nn.functional.batch_norm(
                input,
                running_mean=self.running_mean[:self.num_features],
                running_var=self.running_var[:self.num_features],
                weight=self.weight,
                bias=self.bias,
                training=False,
                momentum=self.momentum,
                eps=self.eps)


class BatchNorm2d(torch.nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1,
                 weight_freeze=False, bias_freeze=False,
                 weight_init=1.0, bias_init=0.0):
        super().__init__(num_features, eps=eps, momentum=momentum)
        if weight_init is not None:
            self.weight.data.fill_(weight_init)
        if bias_init is not None:
            self.bias.data.fill_(bias_init)
        self.weight.requires_grad = not weight_freeze
        self.bias.requires_grad = not bias_freeze


class GhostBatchNorm2d(BatchNorm2d):
    def __init__(self, num_features, num_splits=16, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        self.register_buffer('running_mean',
                             torch.zeros(num_features*self.num_splits))
        self.register_buffer('running_var',
                             torch.ones(num_features*self.num_splits))

    # def train(self, mode=True):
    #     # lazily collate stats when we are going to use them
    #     if (self.training is True) and (mode is False):
    #         self.running_mean = torch.mean(
    #             self.running_mean.view(self.num_splits, self.num_features),
    #             dim=0
    #         ).repeat(self.num_splits)
    #         self.running_var = torch.mean(
    #             self.running_var.view(self.num_splits, self.num_features),
    #             dim=0
    #         ).repeat(self.num_splits)
    #     return super().train(mode)

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            # return torch.nn.functional.batch_norm(
            #     input.view(-1, C*self.num_splits, H, W),
            #     running_mean=self.running_mean,
            #     running_var=self.running_var,
            #     weight=self.weight.repeat(self.num_splits),
            #     bias=self.bias.repeat(self.num_splits),
            #     training=True,
            #     momentum=self.momentum,
            #     eps=self.eps).view(N, C, H, W)
            bn = torch.nn.functional.batch_norm(
                input.view(-1, C*self.num_splits, H, W),
                running_mean=self.running_mean,
                running_var=self.running_var,
                weight=self.weight.repeat(self.num_splits),
                bias=self.bias.repeat(self.num_splits),
                training=True,
                momentum=self.momentum,
                eps=self.eps)
            self.running_mean = torch.mean(
                self.running_mean.view(self.num_splits, self.num_features),
                dim=0
            ).repeat(self.num_splits)
            self.running_var = torch.mean(
                self.running_var.view(self.num_splits, self.num_features),
                dim=0
            ).repeat(self.num_splits)
            return bn.view(N, C, H, W)
        else:
            return torch.nn.functional.batch_norm(
                input,
                running_mean=self.running_mean[:self.num_features],
                running_var=self.running_var[:self.num_features],
                weight=self.weight,
                bias=self.bias,
                training=False,
                momentum=self.momentum,
                eps=self.eps)


def bn_init(bn, scale):
    torch.nn.init.constant_(bn.weight, scale)
    torch.nn.init.constant_(bn.bias, 0)


if __name__ == '__main__':
    gbn1 = GhostBatchNorm1d(10)
    gbn2 = GhostBatchNorm2d(10)
    assert isinstance(gbn1, torch.nn.BatchNorm1d)
    assert isinstance(gbn2, torch.nn.BatchNorm2d)

    bn_init(gbn2, 10.1)
    print(gbn2.weight.data)
