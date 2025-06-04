from typing import Optional

class BaseAgent:
    def generate_response(self, *args, **kwargs) -> str:
        """
        Abstract method to generate a response.
        Subclasses must override this.
        """
        raise NotImplementedError("Subclasses must implement generate_response()")
