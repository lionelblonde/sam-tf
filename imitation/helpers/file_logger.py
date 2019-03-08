from shutil import copyfile
import copy
import os
import yaml
from time import time


class FileLogger:
    """Log to a directory

    Example usage:
        logger = FileLogger({
                                'train': [ 'epoch', 'loss', 'acc' ],
                                'test':  [ 'epoch', 'loss', 'acc' ],
                            },
                            [
                                __file__,
                                'run.py',
                                'setup.sh',
                                'run_baselines.py',
                                'models/layers.py',
                                'models/relational_network.py',
                                'models/resnet.py',
                                'datasets/grid_generator.py',
                                'datasets/grid_loader.py'],
                            path = path,
                            file_prefix = get_short_name(),)

        logger.set_info('note', args.note)
        logger.set_info('uuid', logger.uuid)
        logger.set_info('task', args.task)
        logger.set_info('args', str(args))
    """

    def __init__(self, uuid, log_def={}, source_list=[], path=None,
                 include_wall_time=True, file_prefix=''):
        self.uuid = uuid
        self.dest_path = path
        if not os.path.isdir(path):
            os.makedirs(path)

        self.fp = {}
        self.num_rows = 0
        self.log_def = copy.deepcopy(log_def)
        self.values = {}
        self.info = {}
        self.file_prefix = file_prefix
        self.start_time = None
        if include_wall_time:
            self.start_time = time()

        for log_name, items in log_def.items():
            csv_path = os.path.join(self.dest_path,
                                    file_prefix + self.uuid + "-" + log_name + '.csv')
            print('FILE LOG = ' + csv_path)
            self.fp[log_name] = open(csv_path, 'w')
            self.values[log_name] = {}
            for item in items:
                self.values[log_name][item] = 0.0
        self.source_list = source_list
        self.sources_copied = False

    def copy_sources(self):
        if self.sources_copied:
            return
        self.sources_copied = True
        for path in self.source_list:
            print("copying {}".format(path))
            _, filename = os.path.split(path)
            copyfile(path, os.path.join(self.dest_path, self.uuid + "-" + filename))

    def set_info(self, key, value):  # rewrites yaml file each call
        self.info[key] = value
        with open(os.path.join(self.dest_path, self.uuid + '-info.yml'), 'w') as outfile:
            yaml.dump(self.info, outfile, default_flow_style=False)

    def record(self, log_name, key, value):
        self.values[log_name][key] = value

    def new_row(self, log_name):
        row = str(self.num_rows)
        self.num_rows += 1
        if self.start_time is not None:
            row += ',' + str(time() - self.start_time)
        for key in self.log_def[log_name]:
            row += ',' + str(self.values[log_name][key])
        self.fp[log_name].write(row + "\n")
        self.fp[log_name].flush()
