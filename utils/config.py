import json
from bunch import Bunch
import os

def get_config_from_json(json_file, exp_name = None):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    if type(config_dict) is list:
        idx = 0
        found = False
        if exp_name is None:
            print("No experiment_name specified, using first in config file.")
        else:
            for i in range(len(config_dict)):
                if config_dict[i]['exp_name'] == exp_name:
                    idx = i
                    found = True
                    print("Loading options for exp_name '%s'"%exp_name)
            if not found:
                raise ValueError("Cannot find %s in %s"%(exp_name, json_file))
                #print("Cannot find %s, loading first options in config file"%exp_name)
    else:
        print('Only one experiment found in config file, loading that one.')

    cd = config_dict[idx]
    # convert the dictionary to a namespace using bunch lib
    config = Bunch(cd)

    return config, cd

def process_config(json_file, exp_name = None):
    config, _ = get_config_from_json(json_file, exp_name)
    #Two default values
    config.summary_dir = os.path.join("./experiments", config.exp_name, "summary/")
    config.checkpoint_dir = os.path.join("./experiments", config.exp_name, "checkpoint/")
    return config