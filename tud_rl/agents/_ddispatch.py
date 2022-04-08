import glob
from os.path import dirname, basename, isfile, join
from importlib import import_module
from tud_rl.agents import __path__


modules = glob.glob(join(dirname(__path__[0] + "/discrete/"), "*.py"))
basenames = []
for f in modules:
    if isfile(f) and not f.endswith('__init__.py'):
        basenames.append(basename(f)[:-3])

_DAGENTS = [name + "Agent" for name in basenames]

for name in basenames:
    module = import_module("tud_rl.agents.discrete." + name)
    cls_ = getattr(module, name + "Agent")
    setattr(import_module("tud_rl.agents._ddispatch"), name + "Agent", cls_)
