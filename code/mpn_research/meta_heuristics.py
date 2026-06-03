"""Particle Swarm Optimization (PSO) and Grey Wolf Optimizer (GWO) for hyperparameters."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from . import config
from . import data as data_mod
from .model_builder import build_model


def _decode_particle(vec: np.ndarray) -> tuple[float, float, int]:
    """
    Map a particle in [0,1]^3 to (learning_rate, dropout, batch_size).
    Learning rate is decoded in log-space between LR_MIN and LR_MAX.
    """
    v = np.clip(vec.astype(np.float64), 0.0, 1.0)
    lr = config.LR_MIN * (config.LR_MAX / config.LR_MIN) ** v[0]
    dr = config.DROPOUT_MIN + v[1] * (config.DROPOUT_MAX - config.DROPOUT_MIN)
    batch = config.BATCH_CHOICES[0] if v[2] < 0.5 else config.BATCH_CHOICES[1]
    return float(lr), float(dr), int(batch)


def _train_short(
    lr: float,
    dropout: float,
    batch_size: int,
    epochs: int,
    verbose: int = 0,
) -> float:
    """
    Train a freshly built model for a few epochs; return validation loss (fitness to minimize).
    """
    train_ds, val_ds, _, _ = data_mod.prepare_datasets(batch_size=batch_size)
    model = build_model(learning_rate=lr, dropout_rate=dropout)
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=verbose,
    )
    val_loss = float(hist.history["val_loss"][-1])
    del model
    tf.keras.backend.clear_session()
    return val_loss


def optimize_pso(
    population: int | None = None,
    iterations: int | None = None,
    fitness_epochs: int | None = None,
    verbose: int = 0,
    seed: int | None = None,
) -> tuple[dict, list[float]]:
    """
    Canonical PSO minimizing validation loss.
    Returns (best_hyperparam_dict, global-best validation loss after each generation).
    """
    population = population or config.META_POPULATION
    iterations = iterations or config.META_ITERATIONS
    fitness_epochs = fitness_epochs or config.META_FITNESS_EPOCHS
    seed = seed if seed is not None else config.RANDOM_SEED
    rng = np.random.default_rng(seed)

    dim = 3
    bounds_lo, bounds_hi = np.zeros(dim), np.ones(dim)
    w, c1, c2 = 0.72, 1.49, 1.49
    v_max = 0.2

    X = rng.uniform(bounds_lo, bounds_hi, size=(population, dim))
    V = rng.uniform(-v_max, v_max, size=(population, dim))

    pbest = X.copy()
    pbest_fitness = np.full(population, np.inf)

    history_best: list[float] = []

    for gen in range(iterations):
        gbest_vec = pbest[int(np.argmin(pbest_fitness))].copy()
        for i in range(population):
            if gen > 0:
                r1, r2 = rng.random(dim), rng.random(dim)
                V[i] = (
                    w * V[i]
                    + c1 * r1 * (pbest[i] - X[i])
                    + c2 * r2 * (gbest_vec - X[i])
                )
                V[i] = np.clip(V[i], -v_max, v_max)
                X[i] = np.clip(X[i] + V[i], bounds_lo, bounds_hi)

            lr, dr, bs = _decode_particle(X[i])
            f = _train_short(lr, dr, bs, epochs=fitness_epochs, verbose=verbose)
            if f < pbest_fitness[i]:
                pbest_fitness[i] = f
                pbest[i] = X[i].copy()

        history_best.append(float(np.min(pbest_fitness)))

    gbest_vec = pbest[int(np.argmin(pbest_fitness))]
    lr, dr, bs = _decode_particle(gbest_vec)
    best = {"learning_rate": lr, "dropout": dr, "batch_size": bs}
    return best, history_best


def optimize_gwo(
    population: int | None = None,
    iterations: int | None = None,
    fitness_epochs: int | None = None,
    verbose: int = 0,
    seed: int | None = None,
) -> tuple[dict, list[float]]:
    """
    Grey Wolf Optimizer (Mirjalili et al.) on [0,1]^3 minimizing validation loss.
    """
    population = population or config.META_POPULATION
    iterations = iterations or config.META_ITERATIONS
    fitness_epochs = fitness_epochs or config.META_FITNESS_EPOCHS
    seed = seed if seed is not None else config.RANDOM_SEED
    rng = np.random.default_rng(seed)

    dim = 3
    bounds_lo, bounds_hi = np.zeros(dim), np.ones(dim)

    X = rng.uniform(bounds_lo, bounds_hi, size=(population, dim))

    def evaluate_pack(positions):
        scores = np.empty(len(positions))
        for i, pos in enumerate(positions):
            lr, dr, bs = _decode_particle(pos)
            scores[i] = _train_short(lr, dr, bs, epochs=fitness_epochs, verbose=verbose)
        return scores

    fitness = evaluate_pack(X)
    order = np.argsort(fitness)
    alpha, beta, delta = X[order[0]].copy(), X[order[1]].copy(), X[order[2]].copy()

    history_best = [float(fitness[order[0]])]

    for t in range(max(iterations - 1, 0)):
        # Convergence factor a linearly decreases from ~2 toward 0 across updates
        a = 2.0 * (1.0 - (t + 1) / float(max(iterations, 1)))

        for i in range(population):
            r1, r2 = rng.random(dim), rng.random(dim)
            A1, C1 = 2 * a * r1 - a, 2 * r2
            D_alpha = np.abs(C1 * alpha - X[i])
            X1 = alpha - A1 * D_alpha

            r1, r2 = rng.random(dim), rng.random(dim)
            A2, C2 = 2 * a * r1 - a, 2 * r2
            D_beta = np.abs(C2 * beta - X[i])
            X2 = beta - A2 * D_beta

            r1, r2 = rng.random(dim), rng.random(dim)
            A3, C3 = 2 * a * r1 - a, 2 * r2
            D_delta = np.abs(C3 * delta - X[i])
            X3 = delta - A3 * D_delta

            X[i] = (X1 + X2 + X3) / 3.0
            X[i] = np.clip(X[i], bounds_lo, bounds_hi)

        fitness = evaluate_pack(X)
        order = np.argsort(fitness)
        alpha, beta, delta = X[order[0]].copy(), X[order[1]].copy(), X[order[2]].copy()
        history_best.append(float(fitness[order[0]]))

    lr, dr, bs = _decode_particle(alpha)
    best = {"learning_rate": lr, "dropout": dr, "batch_size": bs}
    return best, history_best


def optimize_hybrid_pso_gwo(
    population: int | None = None,
    iterations: int | None = None,
    fitness_epochs: int | None = None,
    verbose: int = 0,
    seed: int | None = None,
) -> tuple[dict, list[float]]:
    """
    Hybrid meta-heuristic: GWO's α/β/δ pack defines a *global* exploratory direction,
    while PSO-style inertia, velocity, and personal best (pBest) refine each agent.

    For each wolf *i* after the first generation:
      1) Standard GWO encircling toward α, β, δ yields a candidate centroid X_gwo_i.
      2) PSO velocity combines pBest attraction, α-leader attraction, and X_gwo_i attraction:
             V_i ← w V_i + c1 r1 ⊙ (pBest_i − X_i)
                         + c2 r2 ⊙ (α − X_i)
                         + c3 r3 ⊙ (X_gwo_i − X_i)
             X_i ← clip(X_i + V_i)
    This couples exploratory pack dynamics with personal memory to mitigate premature
    convergence on noisy validation loss landscapes.
    """
    population = population or config.META_POPULATION
    iterations = iterations or config.META_ITERATIONS
    fitness_epochs = fitness_epochs or config.META_FITNESS_EPOCHS
    seed = seed if seed is not None else (config.RANDOM_SEED + 101)
    rng = np.random.default_rng(seed)

    dim = 3
    bounds_lo, bounds_hi = np.zeros(dim), np.ones(dim)
    w, c1, c2, c3 = 0.72, 1.35, 1.35, 0.85
    v_max = 0.25

    X = rng.uniform(bounds_lo, bounds_hi, size=(population, dim))
    V = rng.uniform(-v_max, v_max, size=(population, dim))
    pbest = X.copy()
    pbest_fitness = np.full(population, np.inf)

    history_best: list[float] = []

    for gen in range(iterations):
        fitness = np.empty(population, dtype=np.float64)
        for i in range(population):
            lr, dr, bs = _decode_particle(X[i])
            fitness[i] = _train_short(lr, dr, bs, epochs=fitness_epochs, verbose=verbose)
            if fitness[i] < pbest_fitness[i]:
                pbest_fitness[i] = fitness[i]
                pbest[i] = X[i].copy()

        history_best.append(float(np.min(pbest_fitness)))

        if gen == iterations - 1:
            break

        order = np.argsort(fitness)
        alpha = X[order[0]].copy()
        beta = X[order[1]].copy()
        delta = X[order[2]].copy()

        a = 2.0 * (1.0 - gen / float(max(iterations - 1, 1))) if iterations > 1 else 2.0

        for i in range(population):
            r1, r2 = rng.random(dim), rng.random(dim)
            A1, C1 = 2 * a * r1 - a, 2 * rng.random(dim)
            D_alpha = np.abs(C1 * alpha - X[i])
            X1 = alpha - A1 * D_alpha

            r1, r2 = rng.random(dim), rng.random(dim)
            A2, C2 = 2 * a * r1 - a, 2 * rng.random(dim)
            D_beta = np.abs(C2 * beta - X[i])
            X2 = beta - A2 * D_beta

            r1, r2 = rng.random(dim), rng.random(dim)
            A3, C3 = 2 * a * r1 - a, 2 * rng.random(dim)
            D_delta = np.abs(C3 * delta - X[i])
            X3 = delta - A3 * D_delta

            X_gwo = (X1 + X2 + X3) / 3.0

            rp1, rp2, rp3 = rng.random(dim), rng.random(dim), rng.random(dim)
            V[i] = (
                w * V[i]
                + c1 * rp1 * (pbest[i] - X[i])
                + c2 * rp2 * (alpha - X[i])
                + c3 * rp3 * (X_gwo - X[i])
            )
            V[i] = np.clip(V[i], -v_max, v_max)
            X[i] = np.clip(X[i] + V[i], bounds_lo, bounds_hi)

    best_vec = pbest[int(np.argmin(pbest_fitness))]
    lr, dr, bs = _decode_particle(best_vec)
    best = {"learning_rate": lr, "dropout": dr, "batch_size": bs}
    return best, history_best
