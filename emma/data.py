from __future__ import annotations
import torch, random
from torch.utils.data import Dataset, DataLoader
import random

class NeedleDataset(Dataset):
    """
    Multi-pair Needle-in-a-Haystack.

    Sequence (length L) contains n_pairs blocks: <S> K_k V_v <E> at random, non-overlapping positions.
    Query near the end: <Q> K_k ?  with k chosen from the inserted set.
    Target is the *value id v associated with that k in THIS sequence*.

    If decouple_kv is False (legacy), v == k (trivial mapping).
    If decouple_kv is True, (k, v) is chosen independently per pair, per example (requires recall).
    """
    def __init__(
        self,
        num_values: int = 32,
        vocab_extra: int = 64,
        length: int = 256,
        size: int = 2000,
        seed: int = 1234,
        n_pairs: int = 1,
        decouple_kv: bool = True,   # NEW default: require recall
    ):
        super().__init__()
        self.num_values = num_values
        self.vocab_extra = vocab_extra
        self.length = length
        self.size = size
        self.n_pairs = n_pairs
        self.decouple_kv = decouple_kv
        random.seed(seed)

        # Vocab layout
        self.S = vocab_extra
        self.E = vocab_extra + 1
        self.Q = vocab_extra + 2
        self.K0 = vocab_extra + 3
        self.V0 = self.K0 + num_values
        self.vocab_size = self.V0 + num_values

    def __len__(self):
        return self.size

    def _place_block(self, occ, L, block_len=4, low=2, high=None):
        if high is None: high = L - block_len - 1
        for _ in range(200):
            p = random.randrange(low, high)
            if all(not occ[p+i] for i in range(block_len)):
                for i in range(block_len): occ[p+i] = True
                return p
        for p in range(low, high):
            if all(not occ[p+i] for i in range(block_len)):
                for i in range(block_len): occ[p+i] = True
                return p
        raise RuntimeError("Failed to place block; increase length or reduce n_pairs")

    def __getitem__(self, idx: int):
        L = self.length
        seq = [random.randrange(0, self.vocab_extra) for _ in range(L)]
        occ = [False] * L

        # choose distinct keys
        n = min(self.n_pairs, self.num_values)
        keys = random.sample(range(self.num_values), n)

        # choose values (either =keys or random per pair)
        if self.decouple_kv:
            values = [random.randrange(self.num_values) for _ in range(n)]
        else:
            values = list(keys)

        pair_map = {}  # key_id -> value_id
        write_pos_map = {}

        # place pairs in the first 3/4
        for kid, vid in zip(keys, values):
            p0 = self._place_block(occ, L, block_len=4, low=2, high=3*L//4)
            seq[p0]   = self.S
            seq[p0+1] = self.K0 + kid
            seq[p0+2] = self.V0 + vid
            seq[p0+3] = self.E
            pair_map[kid] = vid
            write_pos_map[kid] = p0+2

        # choose which key we query
        qk = random.choice(keys)
        qv = pair_map[qk]

        # place query near end
        p1 = self._place_block(occ, L, block_len=3, low=3*L//4, high=L-3)
        seq[p1]   = self.Q
        seq[p1+1] = self.K0 + qk
        seq[p1+2] = self.vocab_extra  # placeholder '?'

        tokens = torch.tensor(seq, dtype=torch.long)
        key_id = torch.tensor(qk, dtype=torch.long)
        target = torch.tensor(qv, dtype=torch.long)
        write_pos = torch.tensor(write_pos_map[qk], dtype=torch.long)
        query_pos = torch.tensor(p1+2, dtype=torch.long)

        return {
            'tokens': tokens,
            'key_id': key_id,
            'write_pos': write_pos,
            'query_pos': query_pos,
            'target': target,
            'vocab_size': self.vocab_size,
        }

def make_dataloaders(
    num_values=32, vocab_extra=64, length=256, train_size=2000, val_size=200,
    batch_size=32, num_workers=0, seed=1234, n_pairs: int = 1, decouple_kv: bool = True
):
    ds_train = NeedleDataset(num_values, vocab_extra, length, train_size, seed, n_pairs=n_pairs, decouple_kv=decouple_kv)
    ds_val   = NeedleDataset(num_values, vocab_extra, length, val_size, seed+1, n_pairs=n_pairs, decouple_kv=decouple_kv)
    def collate(batch):
        return {
            'tokens': torch.stack([b['tokens'] for b in batch], dim=0),
            'key_id': torch.stack([b['key_id'] for b in batch], dim=0),
            'write_pos': torch.stack([b['write_pos'] for b in batch], dim=0),
            'query_pos': torch.stack([b['query_pos'] for b in batch], dim=0),
            'target': torch.stack([b['target'] for b in batch], dim=0),
            'vocab_size': batch[0]['vocab_size'],
        }
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate, pin_memory=False, persistent_workers=False)
    val_loader   = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate, pin_memory=False, persistent_workers=False)
    return train_loader, val_loader


# ---------------- ListOps-lite (CPU classification test) ----------------
class ListOpsLite(Dataset):
    """Synthetic ListOps-like dataset (MAX/MIN over nested lists).
    Generates prefix-bracketed expressions of digits 0..9; label is MAX over entire structure.
    This is a simplified proxy for quick CPU classification tests at fixed length.
    """
    OPS = {"MAX": max, "MIN": min}

    def __init__(self, length=512, size=1000, seed=1234):
        super().__init__()
        self.length = length
        self.size = size
        random.seed(seed)
        self.op_tokens = ["MAX", "MIN", "[", "]"]

        # Build vocab: 0-9 digits + ops/brackets + PAD
        self.PAD = 0
        self.tok2id = {"<PAD>": self.PAD}
        # digits 0..9 start at 1
        for d in range(10):
            self.tok2id[str(d)] = len(self.tok2id)
        for t in self.op_tokens:
            self.tok2id[t] = len(self.tok2id)
        self.id2tok = {v: k for k, v in self.tok2id.items()}
        self.vocab_size = len(self.tok2id)

    def __len__(self):
        return self.size

    def _gen_expr(self, max_depth=3, max_items=6):
        # Build linear token list directly; compute label as MAX over digits
        def gen_tokens(depth=0):
            if depth >= max_depth or random.random() < 0.4:
                d = random.randint(0, 9)
                return [str(d)], [d]
            op = random.choice("MAX MIN".split())  # op not used in label in lite task
            tokens = ['[', op]
            digits = []
            n_items = random.randint(2, max_items)
            for _ in range(n_items):
                toks, digs = gen_tokens(depth+1)
                tokens.extend(toks)
                digits.extend(digs)
            tokens.append(']')
            return tokens, digits

        expr, digits = gen_tokens(0)
        label = max(digits) if digits else 0
        return expr, label


    def __getitem__(self, idx: int):
        expr, label = self._gen_expr()
        # Map tokens to ids and pad to fixed length
        ids = [self.tok2id.get(t, self.PAD) for t in expr]
        if len(ids) > self.length:
            ids = ids[: self.length]
        if len(ids) < self.length:
            ids = ids + [self.PAD] * (self.length - len(ids))
        tokens = torch.tensor(ids, dtype=torch.long)
        target = torch.tensor(label, dtype=torch.long)
        return {
            'tokens': tokens,
            'target': target,
            'vocab_size': self.vocab_size,
        }

def make_dataloaders_listops_lite(length=512, train_size=2000, val_size=200, batch_size=16, num_workers=0, seed=1234):
    ds_train = ListOpsLite(length=length, size=train_size, seed=seed)
    ds_val   = ListOpsLite(length=length, size=val_size, seed=seed+1)
    def collate(batch):
        return {
            'tokens': torch.stack([b['tokens'] for b in batch], dim=0),
            'target': torch.stack([b['target'] for b in batch], dim=0),
            'vocab_size': batch[0]['vocab_size'],
        }
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate)
    val_loader   = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)
    return train_loader, val_loader
