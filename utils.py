import math
import collections
import torch
import torch.nn as nn


def floatdict2str(dct, precision=4):
    return {
        k: f"{v:.{precision}f}"
        for k, v in dct.items()
    }


def apply_to_sample(f, sample):
    if hasattr(sample, "__len__") and len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, collections.OrderedDict):
            # OrderedDict has attributes that needs to be preserved
            od = collections.OrderedDict((key, _apply(value)) for key, value in x.items())
            od.__dict__ = x.__dict__
            return od
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        elif isinstance(x, tuple):
            return tuple(_apply(x) for x in x)
        elif isinstance(x, set):
            return {_apply(x) for x in x}
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample, device=None):
    device = device or torch.cuda.current_device()

    def _move_to_cuda(tensor):
        # non_blocking is ignored if tensor is not pinned, so we can always set
        # to True (see github.com/PyTorchLightning/pytorch-lightning/issues/620)
        return tensor.to(device=device, non_blocking=True)

    return apply_to_sample(_move_to_cuda, sample)


def multiply_grads(optim, c):
    """Multiplies grads by a constant *c*."""
    for group in optim.param_groups:
        for p in group['params']:
            if p.grad is not None:
                p.grad.data.mul_(c)


class CountingIterator(object):
    """Wrapper around an iterable that maintains the iteration count.
    Args:
        iterable (iterable): iterable to wrap
        start (int): starting iteration count. Note that this doesn't
            actually advance the iterator.
        total (int): override the iterator length returned by ``__len``.
            This can be used to truncate *iterator*.
    Attributes:
        n (int): number of elements consumed from this iterator
    """

    def __init__(self, iterable, start=None, total=None):
        self._itr = iter(iterable)
        self.n = start or getattr(iterable, "n", 0)
        self.total = total or self.n + len(iterable)

    def __len__(self):
        return self.total

    def __iter__(self):
        return self

    def __next__(self):
        if not self.has_next():
            raise StopIteration
        try:
            x = next(self._itr)
        except StopIteration:
            raise IndexError(f"Iterator expected to have length {self.total}, "
                             "but exhausted at position {self.n}.")
        self.n += 1
        return x

    def has_next(self):
        """Whether the iterator has been exhausted."""
        return self.n < self.total

    def skip(self, n):
        """Fast-forward the iterator by skipping n elements."""
        for _ in range(n):
            next(self)
        return self

    def take(self, n):
        """Truncate the iterator to n elements at most."""
        self.total = min(self.total, n)
        # Propagate this change to the underlying iterator
        if hasattr(self._itr, "take"):
            self._itr.take(max(n - self.n, 0))
        return self


class GroupedIterator(CountingIterator):
    """Wrapper around an iterable that returns groups (chunks) of items.
    Args:
        iterable (iterable): iterable to wrap
        chunk_size (int): size of each chunk
    Attributes:
        n (int): number of elements consumed from this iterator
    """

    def __init__(self, iterable, chunk_size):
        itr = _chunk_iterator(iterable, chunk_size)
        super().__init__(
            itr,
            start=int(math.ceil(getattr(iterable, "n", 0) / float(chunk_size))),
            total=int(math.ceil(len(iterable) / float(chunk_size))),
        )
        self.chunk_size = chunk_size


def _chunk_iterator(itr, chunk_size):
    chunk = []
    for x in itr:
        chunk.append(x)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk


class LabelSmoothedCrossEntropyCriterion(nn.Module):
    def __init__(self, smoothing, ignore_index=None, reduce=True):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.reduce = reduce

    def forward(self, lprobs, target):
        if target.dim() == lprobs.dim() - 1:
            target = target.unsqueeze(-1)
        # nll: Negative log likelihood，the cross-entropy when target is one-hot. following line is same as F.nll_loss
        nll_loss = -lprobs.gather(dim=-1, index=target)
        #  reserve some probability for other labels. thus when calculating cross-entropy,
        # equivalent to summing the log probs of all labels
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
        if self.ignore_index is not None:
            pad_mask = target.eq(self.ignore_index)
            nll_loss.masked_fill_(pad_mask, 0.0)
            smooth_loss.masked_fill_(pad_mask, 0.0)
        else:
            nll_loss = nll_loss.squeeze(-1)
            smooth_loss = smooth_loss.squeeze(-1)
        if self.reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        # when calculating cross-entropy, add the loss of other labels
        eps_i = self.smoothing / lprobs.size(-1)
        loss = (1.0 - self.smoothing) * nll_loss + eps_i * smooth_loss
        return loss
