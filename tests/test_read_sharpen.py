import torch

from emma.model import EMMA


def test_read_sharpen_topk_masks_logits():
    logits = torch.tensor([[1.0, 0.9, 0.1, -0.5]])
    sharpened = EMMA._apply_read_sharpen(logits, topk=1, temperature=None, mask_value=-1e9, mask_margin=5.0, training=False, eval_only=False)
    kept_index = torch.argmax(sharpened, dim=-1).item()
    assert torch.allclose(sharpened[0, kept_index], logits[0, kept_index])
    other_indices = [i for i in range(logits.size(1)) if i != kept_index]
    for idx in other_indices:
        assert torch.allclose(sharpened[0, idx], torch.tensor(-1e9, dtype=logits.dtype))


def test_read_sharpen_temperature_scales_logits():
    logits = torch.tensor([[1.0, 0.5]])
    out = EMMA._apply_read_sharpen(logits, topk=None, temperature=0.5, mask_value=-1e9, mask_margin=5.0, training=False, eval_only=False)
    assert torch.allclose(out, logits / 0.5)

