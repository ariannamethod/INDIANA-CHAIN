from ..models import IndianaC, IndianaCConfig


def get_model(config: IndianaCConfig) -> IndianaC:
    """Construct an Indiana-C model from the given configuration."""
    return IndianaC(config)


def get_tokenizer(*args, **kwargs):
    """Tokenizer is intentionally left unimplemented for the Indiana-C core."""
    raise NotImplementedError("Tokenizer support will be added in a later stage.")
