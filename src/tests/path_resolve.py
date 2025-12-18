from sys import path
from os.path import realpath, dirname

# Current
current = dirname(realpath(__file__))
parent = dirname(current)

# Append the parent
path.append(parent)