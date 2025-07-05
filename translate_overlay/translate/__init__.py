from .madlad import MadladTranslator
from .gemma3 import Gemma3Translator
from .gemma3n import Gemma3nTranslator


TRANSLATERS = {
    "MADLAD-400-3B-MT": MadladTranslator,
    "Gemma-3-1B-IT": Gemma3Translator,
    "Gemma-3n-E2B-IT": Gemma3nTranslator,
}
