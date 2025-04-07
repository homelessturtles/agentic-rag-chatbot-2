from crewai.tools import BaseTool
from typing import Type, Optional, Any
from pydantic import BaseModel, Field
try:
    from qdrant_client import QdrantClient

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
import openai
import os
import json


class QdrantToolSchema(BaseModel):
    """Input for Qdrant tool"""
    query: str = Field(..., description="The query to search retrieve relevant information from the Qdrant database. Pass only the query, not the question.")

class QdrantVectorSearchTool(BaseTool):
    model_config = {"arbitrary_types_allowed": True}
    client: QdrantClient = None
    name: str = "QdrantVectorSearchTool"
    description: str = "A tool to search the Qdrant database for relevant information on internal documents."
    args_schema: Type[BaseModel] = QdrantToolSchema
    qdrant_url: str = Field(
        ...,
        description="The URL of the Qdrant server",
    )
    qdrant_api_key: str = Field(
        ...,
        description="The API key for the Qdrant server",
    )
    collection_name: Optional[str] = None
    query: Optional[str] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if QDRANT_AVAILABLE:
            self.client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
            )

    def _run(self, query: str) -> str:

        if not QDRANT_AVAILABLE:
            raise ImportError(
                "The 'qdrant-client' package is required to use the QdrantVectorSearchTool. "
                "Please install it with: pip install qdrant-client"
            )

        if not self.qdrant_url or not self.qdrant_api_key:
            raise ValueError("QDRANT_URL or QDRANT_API_KEY is not set")
        
        search_result = self.client.query(
        collection_name=self.collection_name,
        query_text=query)

        results = []
        # Extract the list of search results
        for response in search_result:
            result = {
                "document": response.metadata['document'],
                "source": response.metadata['source']
            }
            results.append(result)

        return json.dumps(results)
