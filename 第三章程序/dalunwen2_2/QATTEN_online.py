"""
Qatten online inference entry for the disturbed dataset workflow.

This wrapper reuses the existing onlineqmix.py inference loop,
but provides an entry filename that matches the configured mixer.
"""

from config import Config
from onlineqmix import train


if __name__ == "__main__":
    conf = Config()
    print("online entry: QATTEN_online.py, current mixer:", conf.mixer)
    train()
