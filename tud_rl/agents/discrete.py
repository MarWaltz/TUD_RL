"""
Agent dispatcher module. 

This file walks through all `*.py` files in the `_discrete`
folder and sets their classes as attributes of 
the present module.

This allows for a plug-in like system. 

To add a custom agent, create a new file and place
your agent class into it. In order to be recognized by
the plug-in system, your agent class must have the 
same name as your file + "Agent". 

Any class that does not follow this pattern won't be 
an available Agent during training or visualizing.

Example: 
    New file: `CustomDQN.py`
    Class to be exported: `CustomDQNAgent` 
    This is eqiv. to `from CustomDQN import CustomDQNAgent`

"""

import glob
from os.path import dirname, basename, isfile, join
from importlib import import_module
from tud_rl.agents import __path__
from tud_rl import logger

__currentmodule__ = import_module("tud_rl.agents.discrete")

modules = glob.glob(join(dirname(__path__[0] + "/_discrete/"), "*.py"))
_DAGENTS = []
for f in modules:
    if isfile(f) and not f.endswith('__init__.py'):
        _DAGENTS.append(basename(f)[:-3])

logger.info("--- Loading Discrete Agents: ---")

for filename in _DAGENTS:
    module = import_module("tud_rl.agents._discrete." + filename)
    cls_ = getattr(module, filename + "Agent")
    setattr(__currentmodule__, filename + "Agent", cls_)
    logger.info(f"Loading {filename}Agent from {module.__name__}")
