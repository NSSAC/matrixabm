"""Tensorboard Summary Writer.

The tensorboard summary writer
is an actor that is used
to manage access to a tensorboard SummaryWriter object.
This object can then be used to write out stats during runtime.
"""

from tensorboardX import SummaryWriter


class TensorboardWriter:
    """Tensorboard summary writer.

    Attributes
    ----------
    summary_dir : str
        The summary directory
    summary_writer : tensorboardX.SummaryWriter
        The underlying summary writer
    """

    def __init__(self, summary_dir):
        self.summary_dir = summary_dir
        self.summary_writer = SummaryWriter(str(summary_dir))

        self.add_scalar = self.summary_writer.add_scalar
        self.add_histogram = self.summary_writer.add_histogram

        self.flush = self.summary_writer.flush
        self.close = self.summary_writer.close
