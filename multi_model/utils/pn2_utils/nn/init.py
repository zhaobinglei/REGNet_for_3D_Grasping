from torch import nn


def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def set_bn(module, momentum):
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = momentum


def xavier_uniform(module):
    if module.weight is not None:
        nn.init.xavier_uniform_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def xavier_normal(module):
    if module.weight is not None:
        nn.init.xavier_normal_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def kaiming_uniform(module):
    if module.weight is not None:
        nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def kaiming_normal(module):
    if module.weight is not None:
        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
    if module.bias is not None:
        nn.init.zeros_(module.bias)
