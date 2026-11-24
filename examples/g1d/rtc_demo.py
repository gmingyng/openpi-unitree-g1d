"""Standalone inference-time RTC demo. Run with Python + NumPy; all ticks are simulated.

Flow time follows OpenPI: 1 is noise and 0 is clean. The small analytic velocity
field below substitutes for a checkpoint so that its vector-Jacobian product
can be inspected directly. Reference: https://www.physicalintelligence.company/research/real_time_chunking
"""
import numpy as np


HORIZON = 25
DELAY = 7  # 140 ms at 50 Hz
REPLAN = 15
ARM = np.array([*range(7), *range(8, 15)])


def normalize(target, state, q01, q99):
    relative = target.copy()
    relative[:, ARM] -= state[ARM]
    return 2 * (relative - q01) / (q99 - q01 + 1e-6) - 1


def absolute(z, state, q01, q99):
    target = (z + 1) / 2 * (q99 - q01 + 1e-6) + q01
    target[:, ARM] += state[ARM]
    return target


def sample(noise, desired, prefix, delay):
    """Inpaint the old plan in normalized coordinates during ten Euler steps."""
    x = noise.copy()
    overlap = len(prefix)
    if not 0 <= delay <= overlap <= len(x):
        raise ValueError("old plan must cover the inference delay")
    weights = np.zeros((len(x), 1))
    weights[:delay] = 1
    # Simple soft mask: later overlap matters less as the new observation takes over.
    weights[delay:overlap, 0] = np.linspace(1, 0, overlap - delay + 2)[1:-1]
    reference = np.zeros_like(x)
    reference[:overlap] = prefix

    # Toy velocity v(x,t) = K @ x - desired. Temporal coupling makes conditioning
    # one row affect its neighbors. A real flow policy supplies v and autodiff.
    k = np.eye(len(x)) * 0.45 + np.eye(len(x), k=1) * 0.2
    for step in range(10):
        t = 1 - step / 10
        velocity = k @ x - desired
        clean = x - t * velocity
        residual = weights * (reference - clean)
        # J_clean.T @ residual for clean(x) = x - t*v(x,t), holding t fixed.
        correction = residual - t * k.T @ residual
        gain = min(5.0, (t*t + (1-t)**2) / max(t*(1-t), 1e-8))
        x -= 0.1 * (velocity - gain * correction)
    return x


def main():
    rng = np.random.default_rng(7)
    # Synthetic statistics in wire order: L7, Lgrip, R7, Rgrip.
    q01, q99 = np.full(16, -0.3), np.full(16, 0.3)
    q01[[7, 15]], q99[[7, 15]] = 0, 5.4
    state = np.zeros(16)
    # Bootstrap before starting the control clock.
    plan = absolute(sample(rng.normal(size=(HORIZON, 16)), np.zeros((HORIZON, 16)),
                           np.empty((0, 16)), 0), state, q01, q99)
    origin = 0
    pending = None
    for tick in range(60):
        if tick and tick % REPLAN == 0:
            # The request snapshots both state and unexecuted absolute targets.
            old = plan[tick - origin:]
            prefix = normalize(old, state, q01, q99)
            np.testing.assert_allclose(absolute(prefix, state, q01, q99), old, atol=1e-10)
            noise = rng.normal(size=(HORIZON, 16))
            desired = np.full_like(noise, 0.4 * np.sin(tick / 10))
            new = sample(noise, desired, prefix, DELAY)
            baseline = sample(noise, desired, prefix[:0], 0)
            error = lambda z: np.sqrt(np.mean((z[:DELAY] - prefix[:DELAY])**2))
            print(f"request {tick}: overlap={len(old)}, normalized prefix RMSE "
                  f"unguided={error(baseline):.3f}, RTC={error(new):.3f}")
            # Discrete-event simulation of an asynchronous reply.
            pending = tick, absolute(new, state, q01, q99)
        if pending is not None and tick == pending[0] + DELAY:
            origin, plan = pending
            pending = None
            print(f"reply {tick}: switching to new row {tick - origin} (zero-based)")
        # While the request is in flight this still indexes the OLD plan.
        target = plan[tick - origin]
        state += 0.3 * (target - state)  # simulated feedback; no hardware commands


if __name__ == "__main__":
    main()
