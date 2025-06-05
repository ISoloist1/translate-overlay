from abc import ABC, abstractmethod
from typing import Any, List, Dict

class BaseTextRegionDetection(ABC):
    """
    Abstract base class for text region detection tasks.
    """

    @abstractmethod
    def recognize(self, image: Any) -> List[Dict]:
        """
        Perform text region detection on the input image.

        Args:
            image: The input image for text region detection. Can be a file path, numpy array, PIL Image, etc.

        Returns:
            str: The recognized text regions from image.
        """
        
        pass