"""
Simplified improved training functions - uses original data loader with improved config.
This version is more stable and compatible with existing code.
"""
import torch
from lib.train.base_functions import *  # Import all original functions


# Just re-export everything from original base_functions
# The improvements come from the config file (LR schedule, augmentation, etc.)
# Not from changing the data loading logic

def build_dataloaders_simple(cfg, settings):
    """
    Use original dataloader builder - improvements come from config only.
    This is safer and more compatible.
    """
    from lib.train.base_functions import build_dataloaders
    return build_dataloaders(cfg, settings)


# Alias for compatibility
build_dataloaders_hybrid = build_dataloaders_simple

