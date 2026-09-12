"""Learning rate schedulers with warmup and cosine decay."""

import math


class CosineWarmupScheduler:
    def __init__(
        self,
        base_lr: float,
        warmup_steps: int,
        max_steps: int,
        min_lr: float = 1e-6,
    ) -> None:
        self.base_lr = base_lr
        self.warmup_steps = max(warmup_steps, 0)
        self.max_steps = max(max_steps, 1)
        self.min_lr = min_lr

    def get_lr(self, step: int) -> float:
        # 1. Linear warmup
        if step < self.warmup_steps:
            return self.min_lr + (self.base_lr - self.min_lr) * (step / max(1, self.warmup_steps))

        # 2. Constant after max_steps
        if step >= self.max_steps:
            return self.min_lr

        # 3. Cosine decay down to min_lr
        progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine_decay
