"""Get the global summary writer."""

import os
from tensorboardX import SummaryWriter

from . import SUMMARY_DIR_ENVVAR

# Global summary writer object
_SUMMARY_WRITER = None

def get_summary_writer():
    """Get the performace summary writer."""
    global _SUMMARY_WRITER  # pylint: disable=global-statement

    if _SUMMARY_WRITER is not None:
        return _SUMMARY_WRITER

    logdir = os.environ.get(SUMMARY_DIR_ENVVAR, None)
    if not logdir:
        return None

    _SUMMARY_WRITER = SummaryWriter(logdir)
    return _SUMMARY_WRITER
