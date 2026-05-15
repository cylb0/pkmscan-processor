class PokemonCardException(Exception):
    """Base exception for all card processing errors."""

    pass


class PokemonCardDetectionError(PokemonCardException):
    """Raised when the card border cannot be isolated."""

    pass
