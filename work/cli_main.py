import os
from work.unet import unet_model_functions
from work.auxiliary import data_functions
from work.auxiliary.exceptions import *
from work.auxiliary.logger_settings import configure_logger
import click
import logging

LOG_PATH = os.path.abspath('logs')
DATA_PATH = os.path.abspath('data')

log_path = data_functions.create_path(LOG_PATH, 'unet_logs')



@click.group()
def cli():
    """
    Entry function for the command line interface created by Clicky.
    """
    pass


@click.command(name='train_unet', help="train a unet segmentation model,"
                                       "for an example of the parameters dict"
                                       "see the model_training notbook")
@click.option('--params_dict_path', default=None,
              help="path to a json file containing parameters for  .")
@click.option('--src_path', default=None,
              help="path where exsisting model checkpoints and "
                   "parameters exist.")
@click.option('--update_dict_path', default=None,
              help="optional, path to a json file with update parameters for "
                   "loading an exsisting model")
@click.option('--steps', default=None,
              help='Optional, if --src_path is not None, can be used to select'
                   'specific checkpoint to continue training ')
def train_unet(params_dict_path, src_path, update_dict_path, steps):
    configure_logger(name="cherry_stem",
                     console_level='INFO',
                     file_level='INFO',
                     out_path=log_path)

    logger = logging.getLogger(__name__)

    try:
        if params_dict_path is not None:
            if src_path is not None:
                message = 'Ambiguous loading paths, input either' \
                          ' "params_dict_path or "src_path", not both '
                raise UnetModelException(message)
            else:
                params_dict = data_functions.load_json(params_dict_path)
                model = unet_model_functions.ClarifruitUnet(**params_dict)
        else:
            if update_dict_path is not None:
                update_dict = data_functions.load_json(update_dict_path)
            else:
                update_dict = None

            model = unet_model_functions.ClarifruitUnet.load_model(src_path,
                                                                   update_dict,
                                                                   steps)

        keras_logs_path = model.set_model_for_train()
        logger.info(f"for tensorboard use \n "
                    f"tensorboard --logdir={keras_logs_path}")
        model.fit_unet()

    except:
        message = 'Error loading Model'
        logger.exception(message)



if __name__ == '__main__':
    cli.add_command(train_unet)
    cli()