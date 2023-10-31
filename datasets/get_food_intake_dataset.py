import os
import shutil
import csv
import numpy as np
import random
from natsort import os_sorted
import soundfile


headers = ["path","variant","food","sex","other_labels","source"]

def new_row(path,variant,food,sex,other_labels,source):

    return [path,variant,food,sex,other_labels,source]


