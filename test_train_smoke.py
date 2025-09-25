import torch
from emma.model import EMMA
from emma.train import train_epoch
from emma.data import make_dataloaders
from emma.utils import get_device


def build_small_model(device):
    model = EMMA(
        vocab_size=128,
        emb_dim=16,
        hid_dim=32,
        mem_dim=32,
        num_values=16,
        n_slots=64,
        k_top=4,
        deq_max_iter=4,
    )
    return model.to(device)


def test_train_epoch_smoke():
    device = get_device("cpu")
    model = build_small_model(device)

    train_loader, _ = make_dataloaders(
        num_values=16,
        vocab_extra=32,
        length=32,
        train_size=64,
        val_size=16,
        batch_size=8,
        num_workers=0,
        seed=123,
        n_pairs=1,
        decouple_kv=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    loss, acc, *_ = train_epoch(
        model,
        train_loader,
        device,
        optimizer,
        criterion,
        model_type="emma_liquid",
        lam_pred=0.0,
        lam_write=0.0,
        lam_nce=0.0,
        lam_jac=0.0,
    )

    assert loss > 0
    assert 0.0 <= acc <= 1.0

    # Ensure a second epoch still runs (checks that optimizer/model wasn't left in bad state)
    loss2, acc2, *_ = train_epoch(
        model,
        train_loader,
        device,
        optimizer,
        criterion,
        model_type="emma_liquid",
        lam_pred=0.0,
        lam_write=0.0,
        lam_nce=0.0,
        lam_jac=0.0,
    )

    assert loss2 >= 0
    assert 0.0 <= acc2 <= 1.0
