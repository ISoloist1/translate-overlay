from abc import ABC, abstractmethod
from typing import Any, List, Dict

class BaseOCR(ABC):
    """
    Abstract base class for OCR tasks.
    """

    @abstractmethod
    def recognize(self, image: Any) -> List[Dict]:
        """
        Perform OCR on the input image.

        Args:
            image: The input image for OCR. Can be a file path, numpy array, PIL Image, etc.

        Returns:
            str: The recognized text from the image.
        """
        
        pass