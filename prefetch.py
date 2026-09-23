"""Download everything the TP needs from the Hugging Face Hub into the local cache.

Run once per lab machine before the session, so 15 groups don't hit the network at once:

    python prefetch.py
"""

from huggingface_hub import snapshot_download

ITEMS = [
    ("lilkm/pick_cube_franka_panda_30", "dataset"),  # 30 reference demonstrations
    ("lerobot/resnet10", "model"),                   # frozen vision encoder used by the policy
]

for repo_id, repo_type in ITEMS:
    print(f"Downloading {repo_type} {repo_id} ...")
    path = snapshot_download(repo_id, repo_type=repo_type)
    print(f"  -> {path}")

print("\nAll set. Everything is now in the Hugging Face cache (~/.cache/huggingface).")
