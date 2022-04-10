"""
Agent dispatcher module. 

This file walks through all `*.py` files in the `_coninuous`
folder and sets their classes as attributes of 
the present module.

This allows for a plug-in like system. 

To add a custom agent, create a new file and place
your agent class into it. In order to be recognized by
the plug-in system, your agent class must have the 
same name as your file + "Agent". 

Any class that does not follow this pattern won't be 
an available Agent during training.

Example: 
    New file: `CustomDDPG.py`
    Class to be exported: `CustomDDPGAgent`
    This is eqiv. to `from CustomDDPG import CustomDDPGAgent`

"""

import glob
import inspect
from os.path import dirname, basename, isfile, join
from importlib import import_module
from tud_rl.agents import __path__
from tud_rl import logger

__currentmodule__ = import_module("tud_rl.agents.continuous")

modules = glob.glob(join(dirname(__path__[0] + "/_continuous/"), "*.py"))
_CAGENTS = []
for f in modules:
    if isfile(f) and not f.endswith('__init__.py'):
        _CAGENTS.append(basename(f)[:-3])

logger.info("--- Loading Continuous Agents: ---")

for filename in _CAGENTS:
    module = import_module("tud_rl.agents._continuous." + filename)   
    for namestring, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and namestring.endswith("Agent"):
            logger.info("Loading {} from {}.".format(namestring,obj))
            cls_ = getattr(module, filename + "Agent")
            setattr(__currentmodule__, filename + "Agent", cls_)   

