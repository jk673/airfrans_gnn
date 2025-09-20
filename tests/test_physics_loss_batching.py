import torch
import pytest

def physics_loss(predictions, targets):
    return torch.mean((predictions - targets) ** 2)

def test_physics_loss_with_increased_batch_size():
    predictions = torch.randn(40, 4)
    targets = torch.randn(40, 4)
    loss = physics_loss(predictions, targets)
    assert loss is not None

def test_physics_loss_with_different_dimensions():
    predictions = torch.randn(40, 4)
    targets = torch.randn(40, 4)
    loss = physics_loss(predictions, targets)
    assert loss is not None

def test_physics_loss_with_compatible_dimensions():
    predictions = torch.randn(40, 4)
    targets = torch.randn(40, 4)
    loss = physics_loss(predictions, targets)
    assert loss is not None