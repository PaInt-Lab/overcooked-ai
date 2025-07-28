from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
import json
from typing import Dict, Any

class VectorMemory:
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Initialize the embedding model
        self.embedder = HuggingFaceEmbeddings(model_name=embedding_model_name)

        # Create a dummy embedding to find dimensionality
        dummy_embedding = self.embedder.embed_query("dummy")
        dim = len(dummy_embedding)

        # Build an empty FAISS index with that dimension
        index = faiss.IndexFlatL2(dim)
        self.store = FAISS(
            embedding_function=self.embedder,
            index=index,
            docstore=InMemoryDocstore({}),
            index_to_docstore_id={}
        )

    def add_interaction(self, query: str, response: str):
        """
        Store "User: <query>\nBot: <response>" as one document in FAISS.
        """
        text = f"User: {query}\nBot: {response}"
        doc = Document(page_content=text)
        self.store.add_documents([doc])

    def add_game_memory(self, state_summary: Dict[str, Any], action_info: Dict[str, Any], task_title: str):
        """
        Store game state, action information, and task title for later retrieval.
        
        Args:
            state_summary: The state summary from ActionPredictorAgent
            action_info: The action info returned by the agent
            task_title: The task being performed (e.g., "Serving Onion Soup")
        """
        # Create a structured representation for the memory
        memory_text = f"""
Task: {task_title}
State: {json.dumps(state_summary, indent=2)}
Primary Action: {action_info.get('primary_event', 'unknown')}
Secondary Action: {action_info.get('function_call', 'unknown')}
        """.strip()
        
        doc = Document(page_content=memory_text)
        self.store.add_documents([doc])

    def get_game_context(self, current_state: Dict[str, Any], task_title: str, k: int = 5) -> str:
        """
        Return the top-k most similar past game states and actions.
        
        Args:
            current_state: Current state summary to find similar past states
            task_title: Task title to filter by (e.g., "Serving Onion Soup")
            k: Number of similar memories to retrieve
        """
        # Create a query from the current state and task
        query = f"Task: {task_title}\nState: {json.dumps(current_state, indent=2)}"
        
        docs = self.store.similarity_search(query, k=k)
        
        return "\n\n".join(doc.page_content for doc in docs)

    def get_context(self, query: str, k: int = 5) -> str:
        """
        Return the top-k most similar past "User/Bot" interactions as a single string.
        """
        docs = self.store.similarity_search(query, k=k)
        return "\n\n".join(doc.page_content for doc in docs)
