from abc import ABC, abstractmethod

class BaseTranslator(ABC):
    """
    Abstract base class for translation tasks.
    """

    @abstractmethod
    def translate(self, text: str, target_lang: str, source_lang: str) -> str:
        """
        Translate text from source_lang to target_lang.

        :param text: The text to translate.
        :param target_lang: The target language code.
        :param source_lang: The source language code.
        :return: The translated text.
        """

        pass

    
    @abstractmethod
    def source_lang_list(self) -> list:
        """
        Get the list of supported source languages.

        :return: List of supported source languages.
        """

        pass


    @abstractmethod
    def target_lang_list(self) -> list:
        """
        Get the list of supported languages.

        :return: List of supported languages.
        """
        
        pass


    @abstractmethod
    def encode_tokens(self) -> list:
        pass


    @abstractmethod
    def decode_tokens(self) -> str:
        pass
