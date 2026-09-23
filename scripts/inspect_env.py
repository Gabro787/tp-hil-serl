"""Part 1.1: inspect the observation and action spaces of the environment.

    python scripts/inspect_env.py

Prints the spaces, runs 50 random steps, saves the two camera views to runs/plots/cameras.png
and prints the normalisation ranges of the 18-D state (hint for Q1.1).
"""

import gymnasium as gym
import gym_hil  # noqa: F401  (registers the environments)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import PLOTS, REPO, load_reference  # noqa: E402


def show(obs, prefix=""):
    for k, v in obs.items():
        if isinstance(v, dict):
            show(v, prefix + k + ".")
        else:
            print(f"  {prefix + k:28s} shape={getattr(v, 'shape', None)} dtype={getattr(v, 'dtype', type(v).__name__)}")


def main():
    env = gym.make("gym_hil/PandaPickCubeBase-v0", image_obs=True)
    print("Observation space:\n ", env.observation_space)
    print("Action space:\n ", env.action_space)

    obs, info = env.reset(seed=0)
    print("\nObservation content:")
    show(obs)

    rewards = []
    for _ in range(50):
        obs, rew, done, trunc, info = env.step(env.action_space.sample())
        rewards.append(float(rew))
        if done or trunc:
            break
    print(f"\nRewards over {len(rewards)} random steps:\n ", np.round(rewards, 3))
    print("info keys:", list(info.keys()))

    imgs = obs["pixels"]
    fig, axes = plt.subplots(1, len(imgs), figsize=(4 * len(imgs), 4))
    for ax, (name, im) in zip(np.atleast_1d(axes), imgs.items()):
        ax.imshow(im)
        ax.set_title(name)
        ax.axis("off")
    out = PLOTS / "cameras.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"\nCamera views saved to {out.relative_to(REPO)}")
    env.close()

    stats = load_reference("train_config.json")["policy"]["dataset_stats"]["observation.state"]
    print("\nNormalisation ranges of observation.state (hint for Q1.1):")
    print("  idx        min        max")
    for i, (lo, hi) in enumerate(zip(stats["min"], stats["max"])):
        print(f"  {i:3d} {lo:10.3f} {hi:10.3f}")


if __name__ == "__main__":
    main()
