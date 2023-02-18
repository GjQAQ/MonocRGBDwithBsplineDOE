import os

from pytorch_lightning.callbacks.base import Callback


class LogManager(Callback):
    def __init__(self, ckpt_path: str = '', save_when_interrupted: bool = True):
        self.__ckpt_path = ckpt_path
        self.__save_when_interrupted = save_when_interrupted

    def on_init_end(self, trainer):
        if self.__ckpt_path:
            trainer.resume_from_checkpoint = self.__ckpt_path
            print(f'Resum from existing checkpoint: {self.__ckpt_path}')
        else:
            print(f'Starting a new experiment and logging at \n {os.path.expanduser(trainer.logger.log_dir)}')

    def on_keyboard_interrupt(self, trainer, pl_module):
        if self.__save_when_interrupted:
            ckpt_path = os.path.join(trainer.logger.log_dir, 'checkpoints', 'interrupted_model.ckpt')
            trainer.save_checkpoint(ckpt_path)
            print('Saved a checkpoint...')
