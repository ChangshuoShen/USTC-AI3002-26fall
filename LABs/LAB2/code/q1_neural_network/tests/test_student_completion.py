import pytest
import torch

from models import MLP
from utils.data import load_initial_states


@pytest.mark.parametrize("task,group,widths,activation", [
    ("expression", "A", [16], "relu"),
    ("expression", "D", [8, 4], "relu"),
    ("gradient", "B", [32] * 8, "sigmoid"),
])
def test_model_accepts_provided_initial_state(task, group, widths, activation):
    """模型字段和层形状应与课程提供的 state_dict 严格一致。"""
    state = load_initial_states(task)[group]
    model = MLP(2, widths, activation)
    model.load_state_dict(state, strict=True)

    assert set(model.state_dict()) == set(state)
    for name, value in model.state_dict().items():
        assert tuple(value.shape) == tuple(state[name].shape)
        torch.testing.assert_close(value, state[name])
