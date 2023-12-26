import argparse
import os

import pandas as pd
import preprocess
import result
import tensorflow as tf
import training
import utils
import visualize
import keras
from pathlib import Path

MODEL_DIR = "models"

STORE = "store"
STORE_CONST = "store_const"


parser = argparse.ArgumentParser(description="UB-NSSD YAMNet transfer learning model to transcribe audio (only-intake)")

parser.add_argument("audio_path",
                    help="Audio file to transcribe")

# can get the sr from audio, so unsure if necessary
# parser.add_argument("--sr", action="store_const") 

# window size
parser.add_argument("--size", action=STORE_CONST, const=0.98,
                    help="Size of prediction window, in seconds, default=0.98")
# step size
parser.add_argument("--step", action=STORE_CONST, const=0.10, 
                    help="Step to move window, in seconds, default=0.10")

parser.add_argument("--save_path", action=STORE,
                    help="file path to store the graph")

parser.add_argument("--version", action=STORE_CONST, const="only_intake",
                    help="Version of the model to transcribe with. \
                        [only_intake, all], default=only_intake")


args = parser.parse_args()

model_path = Path("models",args.version)