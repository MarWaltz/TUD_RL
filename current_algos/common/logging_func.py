"""
Some simple logging functionality, inspired by rllab's logging.
Logs to a tab-separated-values file (path/to/output_directory/progress.txt)
"""
import json
import numpy as np
from datetime import date
import os.path as osp
import time
import atexit
import os
from mpi4py import MPI
import numpy as np


def allreduce(*args, **kwargs):
    return MPI.COMM_WORLD.Allreduce(*args, **kwargs)


def num_procs():
    """Count active MPI processes."""
    return MPI.COMM_WORLD.Get_size()


def mpi_op(x, op):
    x, scalar = ([x], True) if np.isscalar(x) else (x, False)
    x = np.asarray(x, dtype=np.float32)
    buff = np.zeros_like(x, dtype=np.float32)
    allreduce(x, buff, op=op)
    return buff[0] if scalar else buff


def mpi_sum(x):
    return mpi_op(x, MPI.SUM)


def mpi_avg(x):
    """Average a scalar or vector over MPI processes."""
    return mpi_sum(x) / num_procs()


def mpi_statistics_scalar(x, with_min_and_max=False):
    """
    Get mean/std and optional min/max of scalar x across MPI processes.
    Args:
        x: An array containing samples of the scalar to produce statistics
            for.
        with_min_and_max (bool): If true, return min and max of x in 
            addition to mean and std.
    """
    x = np.array(x, dtype=np.float32)
    global_sum, global_n = mpi_sum([np.sum(x), len(x)])
    mean = global_sum / global_n

    global_sum_sq = mpi_sum(np.sum((x - mean)**2))
    std = np.sqrt(global_sum_sq / global_n)  # compute global std

    if with_min_and_max:
        global_min = mpi_op(np.min(x) if len(x) > 0 else np.inf, op=MPI.MIN)
        global_max = mpi_op(np.max(x) if len(x) > 0 else -np.inf, op=MPI.MAX)
        return mean, std, global_min, global_max
    return mean, std


def is_json_serializable(v):
    try:
        json.dumps(v)
        return True
    except:
        return False


def convert_json(obj):
    """ Convert obj to a version which can be serialized with JSON. """
    if is_json_serializable(obj):
        return obj
    else:
        if isinstance(obj, dict):
            return {convert_json(k): convert_json(v)
                    for k, v in obj.items()}

        elif isinstance(obj, tuple):
            return (convert_json(x) for x in obj)

        elif isinstance(obj, list):
            return [convert_json(x) for x in obj]

        elif hasattr(obj, '__name__') and not('lambda' in obj.__name__):
            return convert_json(obj.__name__)

        elif hasattr(obj, '__dict__') and obj.__dict__:
            obj_dict = {convert_json(k): convert_json(v)
                        for k, v in obj.__dict__.items()}
            return {str(obj): obj_dict}

        return str(obj)


class Logger:
    """
    A general-purpose logger.
    Makes it easy to save diagnostics, hyperparameter configurations, the 
    state of a training run, and the trained model.
    """

    def __init__(self, alg_str, output_dir=None, output_fname='progress.txt', exp_name=None):
        """
        Initialize a Logger.
        Args:
            output_dir (string): A directory for saving results to. If 
                ``None``, defaults to a temp directory of the form
                ``/tmp/experiments/somerandomnumber``.
            output_fname (string): Name for the tab-separated-value file 
                containing metrics logged throughout a training run. 
                Defaults to ``progress.txt``. 
            exp_name (string): Experiment name. If you run multiple training
                runs and give them all the same ``exp_name``, the plotter
                will know to group them. (Use case: if you run the same
                hyperparameter configuration with multiple random seeds, you
                should give them all the same ``exp_name``.)
        """

        # Scan through a list of dir strings to strip the ascending number
        def max_number(dir_list):
            numbers = []
            if not dir_list:
                return 0
            else:
                for dir in dir_list:
                    numbers.append(int(dir.split("_")[-1]))
            return max(numbers)

         # create output directory
        if output_dir is not None:
            self.output_dir = output_dir
        else:

            if not osp.exists("experiments"):
                os.makedirs("experiments")

            # Get date from today and format to "yyyy-mm-dd_"
            today = date.today().strftime("%Y-%m-%d_")
            # Get folder names in experiments folder
            folder_names = [f.name for f in os.scandir(
                "experiments") if f.is_dir()]
            folder_from_today = [name for name in folder_names if name.startswith(alg_str + "_" + today)]  # Folders starting with todays date
            count = max_number(folder_from_today)
            count += 1
            random_str = str(time.time())[-3:]

            self.output_dir = "experiments/" + alg_str + \
                "_" + today + random_str + "_" + str(count)

        os.makedirs(self.output_dir)

        # create output file and automated closing when file terminates
        self.output_file = open(osp.join(self.output_dir, output_fname), 'w')
        print(f"Logging data to {self.output_file.name}")
        atexit.register(self.output_file.close)

        self.first_row = True
        self.log_headers = []
        self.log_current_row = {}
        self.exp_name = exp_name

    def log_tabular(self, key, val):
        """
        Log a value of some diagnostic.
        Call this only once for each diagnostic quantity, each iteration.
        After using ``log_tabular`` to store values for each diagnostic,
        make sure to call ``dump_tabular`` to write them out to file and
        stdout (otherwise they will not get saved anywhere).
        """
        if self.first_row:
            self.log_headers.append(key)
        else:
            assert key in self.log_headers, "Trying to introduce a new key %s that you didn't include in the first iteration" % key
        assert key not in self.log_current_row, "You already set %s this iteration. Maybe you forgot to call dump_tabular()" % key
        self.log_current_row[key] = val

    def save_config(self, config):
        """
        Log an experiment configuration.
        Call this once at the top of your experiment, passing in all important
        config vars as a dict. This will serialize the config to JSON, while
        handling anything which can't be serialized in a graceful way (writing
        as informative a string as possible). 
        Example use:
        .. code-block:: python
            logger = EpochLogger(**logger_kwargs)
            logger.save_config(locals())
        """
        config_json = convert_json(config)
        if self.exp_name is not None:
            config_json['exp_name'] = self.exp_name

        output = json.dumps(config_json, separators=(
            ',', ':\t'), indent=4, sort_keys=True)
        print('Saving config:\n')
        print(output)
        with open(osp.join(self.output_dir, "config.json"), 'w') as out:
            out.write(output)

    def dump_tabular(self):
        """
        Write all of the diagnostics from the current iteration.
        Writes both to stdout, and to the output file.
        """
        vals = []
        key_lens = [len(key) for key in self.log_headers]
        max_key_len = max(15, max(key_lens))
        keystr = '%' + '%d' % max_key_len
        fmt = "| " + keystr + "s | %15s |"
        n_slashes = 22 + max_key_len
        print("-" * n_slashes)
        for key in self.log_headers:
            val = self.log_current_row.get(key, "")
            valstr = "%8.3g" % val if hasattr(val, "__float__") else val
            print(fmt % (key, valstr))
            vals.append(val)
        print("-" * n_slashes, flush=True)
        if self.output_file is not None:
            if self.first_row:
                self.output_file.write("\t".join(self.log_headers) + "\n")
            self.output_file.write("\t".join(map(str, vals)) + "\n")
            self.output_file.flush()
        self.log_current_row.clear()
        self.first_row = False


class EpochLogger(Logger):
    """
    A variant of Logger tailored for tracking average values over epochs.
    Typical use case: there is some quantity which is calculated many times
    throughout an epoch, and at the end of the epoch, you would like to 
    report the average / std / min / max value of that quantity.
    With an EpochLogger, each time the quantity is calculated, you would
    use 
    .. code-block:: python
        epoch_logger.store(NameOfQuantity=quantity_value)
    to load it into the EpochLogger's state. Then at the end of the epoch, you 
    would use 
    .. code-block:: python
        epoch_logger.log_tabular(NameOfQuantity, **options)
    to record the desired values.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_dict = dict()

    def store(self, **kwargs):
        """
        Save something into the epoch_logger's current state.
        Provide an arbitrary number of keyword arguments with numerical 
        values.
        """
        for k, v in kwargs.items():
            if not(k in self.epoch_dict.keys()):
                self.epoch_dict[k] = []
            self.epoch_dict[k].append(v)

    def log_tabular(self, key, val=None, with_min_and_max=False, average_only=False):
        """
        Log a value or possibly the mean/std/min/max values of a diagnostic.
        Args:
            key (string): The name of the diagnostic. If you are logging a
                diagnostic whose state has previously been saved with 
                ``store``, the key here has to match the key you used there.
            val: A value for the diagnostic. If you have previously saved
                values for this key via ``store``, do *not* provide a ``val``
                here.
            with_min_and_max (bool): If true, log min and max values of the 
                diagnostic over the epoch.
            average_only (bool): If true, do not log the standard deviation
                of the diagnostic over the epoch.
        """
        if val is not None:
            super().log_tabular(key, val)
        else:
            v = self.epoch_dict[key]
            vals = np.concatenate(v) if isinstance(
                v[0], np.ndarray) and len(v[0].shape) > 0 else v
            stats = mpi_statistics_scalar(
                vals, with_min_and_max=with_min_and_max)
            super().log_tabular(
                key if average_only else 'Avg_' + key, stats[0])
            if not(average_only):
                super().log_tabular('Std_' + key, stats[1])
            if with_min_and_max:
                super().log_tabular('Max_' + key, stats[3])
                super().log_tabular('Min_' + key, stats[2])
        self.epoch_dict[key] = []


"""
x = 5
a = 3
import numpy as np

logger = EpochLogger()
logger.save_config(locals())

N_EPISODES = 10
STEPS_PER_EPISODE = 20

for e in range(N_EPISODES):
    
    for t in range(STEPS_PER_EPISODE):

        logger.store(Actor_loss=np.random.rand(1))
        logger.store(Critic_loss=np.random.rand(1))
    
    logger.log_tabular("Actor_loss", with_min_and_max=True)
    logger.log_tabular("Critic_loss", with_min_and_max=False)
    logger.dump_tabular()
    
"""
