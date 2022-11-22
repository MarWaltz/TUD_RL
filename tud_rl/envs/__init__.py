import glob
import inspect
from os.path import dirname, basename, isfile, join
from importlib import import_module
from tud_rl.envs import __path__
from tud_rl import logger

try:
    import gym_minatar
except ImportError as e:
    logger.warning(
        f"Error while importing gym_minatar: {e.msg}. Skipping..."
    )

__currentmodule__ = import_module("tud_rl.envs")

modules = glob.glob(join(dirname(__path__[0] + "/_envs/"), "*.py"))
_ENVS = []
for f in modules:
    if isfile(f) and not f.endswith('__init__.py'):
        _ENVS.append(basename(f)[:-3])

logger.info("--- Loading Environments: ---")

for filename in _ENVS:
    try:
        module = import_module("tud_rl.envs._envs." + filename)
    except ImportError as e:
        logger.warning(
            f"Error while importing {filename}: {e.msg}. Skipping..."
        )
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and all(hasattr(obj, a) for a in ["reset", "step"]):
            cls_ = getattr(module, name)
            logger.info(f"Loading {name} from {module.__name__}.")
            setattr(__currentmodule__, name, cls_)