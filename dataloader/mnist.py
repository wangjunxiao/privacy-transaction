from json import loads
import numpy as np
from base64 import b64decode

def parse(x):
    """
    to parse the digits file into tuples of 
    (labelled digit, numpy array of vector representation of digit)
    """
    digit = loads(x)
    array = np.frombuffer(b64decode(digit["data"]), dtype=np.ubyte)
    array = array.astype(np.float64)
    return (digit["label"], array)

def loader():
    # read in the digits file. Digits is a list of 60,000 tuples,
    # each containing a labelled digit and its vector representation.
    labeled_dataset = []
    with open("dataset/digits.base64.json", "r") as f:
        lines = f.readlines()
        for line in lines:
            labeled_dataset.append(parse(line))
    labels = np.array([x[0] for x in labeled_dataset])
    data = np.array([x[1] for x in labeled_dataset])
    return labels, data