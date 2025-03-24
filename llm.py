"""
A module for interacting with OpenAI's GPT models through a wrapper class.
"""

import os
from typing import Any, Dict

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class GPT:
    """
    A wrapper class for interacting with OpenAI's GPT models.

    Attributes:
        api_key (str): The API key for authentication with OpenAI.
        organization_key (str, optional): The organization ID for OpenAI.
        client (OpenAI): The OpenAI client instance.
    """

    def __init__(
        self,
        api_key: str = None,
        organization_key: str = None
    ) -> None:
        """
        Initializes the GPT class with OpenAI credentials.
        Args:
            api_key (str, optional): OpenAI API key. Defaults to None, in which case
                it is retrieved from environment variables.
            organization_key (str, optional): OpenAI organization ID. Defaults to None,
                retrieved from environment variables.

        Raises:
            ValueError: If no API key is provided.
        """
        api_key = os.getenv("OPENAI_API_KEY", api_key)
        organization_key = os.getenv("OPENAI_ORGANIZATION_ID", organization_key)

        if not api_key:
            raise ValueError(
                "An API key must be provided either as an argument or in the "
                "environment variable 'OPENAI_API_KEY'. This is the recommended approach."
            )

        self.api_key = api_key
        self.organization_key = organization_key
        self.client = OpenAI(
            api_key=self.api_key,
            organization=self.organization_key
        )

    def get_response(self, **kwargs: Dict[str, Any]) -> Any:
        """Sends a request to the OpenAI API and returns the response."""
        response = self.client.responses.create(**kwargs)
        return response
