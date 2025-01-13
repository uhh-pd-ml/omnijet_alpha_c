from lightning.pytorch.callbacks import ModelCheckpoint


class CustomModelCheckpoint(ModelCheckpoint):
    """Custom ModelCheckpoint callback that allows to specify the state_key to be used for the best
    checkpoint.

    This workaround is needed because it's not allowed to have two ModelCheckpoint callbacks with
    the same state_key in the same Trainer.
    """

    def __init__(self, state_key="best_checkpoint", **kwargs):
        super().__init__(**kwargs)
        self._state_key = state_key

    @property
    def state_key(self) -> str:
        return self._state_key
