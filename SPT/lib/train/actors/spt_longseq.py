from .spt import SPTActor


class SPTLongSeqActor(SPTActor):
    """
    Compatibility wrapper around SPTActor for long-sequence training.
    The base actor now natively supports sequential search frames, so this class simply reuses it.
    """
    pass
