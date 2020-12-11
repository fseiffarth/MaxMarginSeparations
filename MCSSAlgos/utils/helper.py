from logging import Logger

import yaml


def load_params(path: str, logger: Logger) -> dict:
    """Loads experiment parameters from json file.

    :param path: to the json file
    :param logger:
    :returns: param needed for the experiment
    :rtype: dictionary

    """
    try:
        with open(path, "rb") as f:
            params = yaml.full_load(f)
        return params
    except Exception as e:
        logger.error(e)