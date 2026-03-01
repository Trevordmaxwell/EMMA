import torch
from emma.model import EMMA


def make_model():
    return EMMA(
        vocab_size=128,
        emb_dim=32,
        hid_dim=64,
        mem_dim=64,
        num_values=16,
        n_slots=64,
        k_top=8,
    ).to(torch.device("cpu"))


def test_memory_resets_between_forwards():
    model = make_model()

    tokens = torch.randint(0, 128, (2, 32))
    key_ids = torch.randint(0, 16, (2,))
    write_pos = torch.randint(0, 32, (2,))
    query_pos = torch.randint(0, 32, (2,))
    value_ids = torch.randint(0, 16, (2,))

    _, metrics = model(tokens, key_ids, write_pos, query_pos, value_ids=value_ids)
    assert metrics["num_write_steps"] >= 0
    assert model.memory.memory.abs().sum() > 0

    # Second pass with writes disabled: if reset happens, memory should remain zero.
    _, metrics_disabled = model(
        tokens,
        key_ids,
        write_pos,
        query_pos,
        value_ids=value_ids,
        disable_write=True,
    )
    assert metrics_disabled["num_write_steps"] >= 0
    assert torch.allclose(model.memory.memory, torch.zeros_like(model.memory.memory))
