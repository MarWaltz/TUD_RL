import glob
from os.path import dirname, basename, isfile, join
from importlib import import_module
from typing import List


modules = glob.glob(join(dirname(__file__), "*.py"))
basenames = []
for f in modules:
    if isfile(f) and not f.endswith('__init__.py'):
        basenames.append(basename(f)[:-3])

agents: List = []
for name in basenames:
    module = import_module(name)
    agents.append(getattr(module, name + "Agent"))