"""
Qatten training entry for the disturbed dataset workflow.

This wrapper keeps the original training loop in QMIX_dis1.py,
but makes the entry filename match the configured mixer.
"""

from config import Config
from QMIX_dis1 import train


if __name__ == "__main__":
    conf = Config()
    print(
        "training entry: QATTEN_dis1.py, current mixer:",
        conf.mixer,
        "model_tag:",
        conf.model_tag,
    )
    train()
