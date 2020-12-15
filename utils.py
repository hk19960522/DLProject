import logging
import os
import yaml
import torch
import numpy as np


def write_result(loss_series, ADE_series, FDE_series):
    file = open('Result.txt', 'w')
    for idx in range(len(loss_series)):
        file.write(f'{idx} {loss_series[idx]} {ADE_series[idx]} {FDE_series[idx]}\n')

    file.close()


