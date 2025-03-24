import traceback

# import docker
import qdrant_client
import numpy as np

from tqdm import tqdm
from typing import List, Dict, Tuple, Literal, Any

from qdrant_client import models
from dotenv import load_dotenv

load_dotenv()

class QdrantVectorDatabase:
    def __init__(self, qdrant_client_instance, openai_client=None, sentence_transformer_encoder=None):
        """
        Initialize the QdrantVectorDatabase instance.

        Args:
            qdrant_client_instance (qdrant_client.QdrantClient): Instance of the Qdrant client.
            openai_client (Any, optional): The OpenAI client instance for generating vectors using OpenAI models. Defaults to None.
            sentence_transformer_encoder (Any, optional): The Sentence Transformers encoder instance for generating vectors using Sentence Transformers models. Defaults to None.
        """
        self.qdrant_client_instance = qdrant_client_instance
        self.openai_client = openai_client
        self.sentence_transformer_encoder = sentence_transformer_encoder

    def generate_vector(self, sentence: str, model_type: Literal["openai", "sentence_transformer"], **kwargs) -> np.ndarray:
        """
        Generates a vector for a single sentence using either OpenAI embeddings or Sentence Transformers.

        Args:
            sentence (str): The sentence to be converted into a vector.
            model_type (Literal["openai", "sentence_transformer"]): The type of model to use for generating the vector.

        Keyword Args:
            openai_model (str, optional): The specific OpenAI model to use if model_type is "openai".

        Returns:
            np.ndarray: The generated vector as a numpy array.
        
        Raises:
            ValueError: If the 'openai_model' is not provided when using OpenAI model type.
        """
        if model_type == "openai":
            openai_model = kwargs.get("openai_model", None)
            if not openai_model:
                raise ValueError("For model_type 'openai', you must pass the 'openai_model' name.")

        if model_type == 'openai':
            try:
                response = self.openai_client.embeddings.create(input=sentence, model=openai_model)
                return np.array(response.data[0].embedding)
            except Exception as e:
                print(f"Error generating OpenAI embedding: {e}")
                return np.array([])
        elif model_type == 'sentence_transformers':
            try:
                return self.sentence_transformer_encoder.encode([sentence], show_progress_bar=False)[0]
            except Exception as e:
                print(f"Error generating Sentence Transformer embedding: {e}")
                return np.array([])
        else:
            print("Invalid `model_type` valid options ['openai', 'sentence-transformers']")

    def generate_vectors_batch(self, sentences: List[str], model_type: Literal["openai", "sentence_transformer"], **kwargs) -> np.ndarray:
        """
        Generates vectors for a list of sentences using either OpenAI embeddings or Sentence Transformers.

        Args:
            sentences (List[str]): The list of sentences to be converted into vectors.
            model_type (Literal["openai", "sentence_transformer"]): The type of model to use for generating the vectors.

        Keyword Args:
            openai_model (str, optional): The specific OpenAI model to use if model_type is "openai".
            max_sentences_per_call (int, optional): The maximum number of sentences to process in a single call. Defaults to 2048.

        Returns:
            np.ndarray: The generated vectors as a numpy array.
        
        Raises:
            ValueError: If the 'openai_model' is not provided when using OpenAI model type.
        """
        if model_type == "openai":
            openai_model = kwargs.get("openai_model", None)
            if not openai_model:
                raise ValueError("For model_type 'openai', you must pass the 'openai_model' name.")

        max_sentences_per_call = kwargs.get("max_sentences_per_call", 2048)
        
        all_embeddings = []        
        num_chunks = len(sentences) // max_sentences_per_call + (1 if len(sentences) % max_sentences_per_call > 0 else 0)

        for i in tqdm(range(num_chunks), desc="Embedding"):
            start_idx = i * max_sentences_per_call
            end_idx = start_idx + max_sentences_per_call
            current_chunk = sentences[start_idx:end_idx]

            if model_type == 'openai':
                try:
                    if openai_model is None:
                        raise ValueError("Invalid openai model name!")
                    response = self.openai_client.embeddings.create(input=current_chunk, model=openai_model)
                    embeddings = [embedding.embedding for embedding in response.data]
                    all_embeddings.extend(embeddings)
                except Exception as e:
                    print(f"Something wrong with the OpenAI embeddings chunk from index {start_idx} to {end_idx}: {e}")
                    return np.array([])
            elif model_type == 'sentence_transformer':
                try:
                    embeddings = self.sentence_transformer_encoder.encode(current_chunk, show_progress_bar=False)
                    all_embeddings.append(embeddings)
                except Exception as e:
                    print(f"Something wrong with the Sentence Transformers embeddings chunk from index {start_idx} to {end_idx}: {e}")
                    return np.array([])

        return np.vstack(all_embeddings)

    def create_qdrant_collection(self, collection_name: str, vector_shape: int):
        """
        Create a Qdrant collection if it does not already exist.

        Args:
            collection_name (str): The name of the collection to be created.
            vector_shape (int): The dimensionality of the vectors to be stored in the collection.
        """
        collection_exists = self.qdrant_client_instance.collection_exists(collection_name=collection_name)
        
        if collection_exists:
            print(f"Collection '{collection_name}' already exists!")
        else:
            print(f"Collection '{collection_name}' not found! Attempting to create...")
            try:
                self.qdrant_client_instance.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_shape, 
                        distance=models.Distance.COSINE, 
                        on_disk=True
                    )
                )
                print(f"Collection '{collection_name}' created!")
            except Exception as e:
                print(f"Failed to create collection '{collection_name}': {e}")

    def delete_qdrant_collection(self, collection_name: str):
        """
        Delete a Qdrant collection.

        Args:
            collection_name (str): The name of the collection to be deleted.
        """
        collection_exists = self.qdrant_client_instance.collection_exists(collection_name=collection_name)
        
        if not collection_exists:
            print(f"Collection '{collection_name}' doesn't exist.")
        else:
            self.qdrant_client_instance.delete_collection(collection_name)
            print(f"Collection '{collection_name}' deleted.")

    def get_collection(self, collection_name: str) -> Dict:
        """
        Get the configuration of a Qdrant collection.

        Args:
            collection_name (str): The name of the collection.

        Returns:
            Dict: The configuration of the collection.
        """
        collection_exists = self.qdrant_client_instance.collection_exists(collection_name=collection_name)
        
        if not collection_exists:
            print(f"Collection '{collection_name}' doesn't exist.")
            return {}
        
        collection_config = dict(self.qdrant_client_instance.get_collection(collection_name))
        return collection_config

    def get_collections(self) -> Dict:
        """
        Get the configuration of a Qdrant collection.

        Args:
            collection_name (str): The name of the collection.

        Returns:
            Dict: The configuration of the collection.
        """
        all_collections = self.qdrant_client_instance.get_collections()
        collection_list = [collection.name for collection in all_collections.collections]
        return collection_list
        
    def upsert_in_chunks(self, collection_name: str, ids: List[str], payloads: List[dict], vectors: List[List[float]], chunk_size: int = 1000):
        """
        Upserts data into the collection in chunks to avoid write timeouts.

        Args:
            collection_name (str): The name of the collection to upsert data into.
            ids (List[str]): The list of IDs for the points.
            payloads (List[dict]): The list of payloads for the points.
            vectors (List[List[float]]): The list of vectors for the points.
            chunk_size (int, optional): The size of each chunk for upserting data. Defaults to 1000.
        """
        total_chunks = len(vectors) // chunk_size + (1 if len(vectors) % chunk_size > 0 else 0)

        for i in tqdm(range(total_chunks), desc="Uploading Chunks"):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size

            ids_chunk = ids[start_idx:end_idx]
            payloads_chunk = payloads[start_idx:end_idx]
            vectors_chunk = vectors[start_idx:end_idx]

            self.qdrant_client_instance.upsert(
                collection_name=collection_name,
                points=qdrant_client.models.Batch(
                    ids=ids_chunk,
                    payloads=payloads_chunk,
                    vectors=vectors_chunk,
                ),
            )

    def search_vectors(self, collection_name: str, query_vector: List[float], limit: int = 10) -> Any:
        """
        Performs a vector search in the specified collection.

        Args:
            collection_name (str): The name of the collection to search in.
            query_vector (List[float]): The query vector to search for.
            limit (int, optional): The number of top results to return. Defaults to 10.

        Returns:
            Any: The search results from the collection.
        """
        result = self.qdrant_client_instance.search(collection_name=collection_name, query_vector=query_vector, limit=limit)
        return result

    def find_top_matches(self, collection_name: str, sentence: str, model_type: str="openai", limit: int=10, **kwargs) -> List:
        """
        Finds the top matching vectors in a specified collection based on a query vector generated from a sentence.

        Args:
            collection_name (str): The name of the collection to search in.
            sentence (str): The sentence to generate the query vector from.
            model_type (str, optional): The type of model to use for generating the vector. Defaults to "openai".
            limit (int, optional): The number of top results to return. Defaults to 10.

        Keyword Args:
            openai_model (str, optional): The specific OpenAI model to use if model_type is "openai".

        Returns:
            List: The list of top matching vectors from the collection.

        Raises:
            ValueError: If the 'openai_model' is not provided when using OpenAI model type.
        """
        if model_type == "openai":
            openai_model = kwargs.get("openai_model", None)
            if not openai_model:
                raise ValueError("For model_type 'openai', you must pass the 'openai_model' name.")

        try:
            query_vector = self.generate_vector(sentence=sentence.lower(), model_type=model_type, **kwargs)
            match_vectors = self.search_vectors(collection_name=collection_name, query_vector=query_vector, limit=limit)
            return match_vectors
        except Exception as e:
            print(f"Some unknown exception occurred! {e}")
            print(traceback.format_exc())
            return []
