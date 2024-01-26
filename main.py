from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores import WeaviateVectorStore
from llama_index.storage.storage_context import StorageContext
import weaviate
from llama_index.llms import Anyscale
from llama_index import ServiceContext
from llama_index.postprocessor import CohereRerank
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.postprocessor import LLMRerank
from typing import Any, List, Optional
from pydantic import BaseModel
import aiohttp
from fastapi.responses import StreamingResponse

class QueryModel(BaseModel):
    query: str

class QueryResponseModel(BaseModel):
    success: bool
    data: dict  # or a more specific model if you have a fixed structure
    error: Optional[str] = None

# Votre classe VectorStoreQueryEngine et les autres imports nécessaires ici...
class VectorStoreQueryEngine:
    def __init__(self, weaviate_url, api_key, service_context):
        # Initialisation du client Weaviate avec les paramètres de configuration
        resource_owner_config = weaviate.AuthApiKey(api_key=api_key)
        self.client = weaviate.Client(
            weaviate_url,
            auth_client_secret=resource_owner_config,
        )
        self.service_context = service_context

    def _get_query_engine(self, index_name):
        vector_store = WeaviateVectorStore(
            weaviate_client=self.client, index_name=index_name
        )
        loaded_index = VectorStoreIndex.from_vector_store(vector_store)

        query_engine = loaded_index.as_query_engine(
            service_context=self.service_context, 
            verbose=True,
            similarity_top_k=10
        )
        return query_engine
    def query_llama_index(self, query):
        query_engine = self._get_query_engine("LlamaIndex")
        response = query_engine.query(query)
        response.print_response_stream()
        return response.text

    def query_llama_hub(self, query):
        query_engine = self._get_query_engine("LLamaHubCode")
        response = query_engine.query(query)
        print("Response from query engine:", response)
        return str(response)
        
    def query_llama_lab(self, query):
        query_engine = self._get_query_engine("LLamaLabCode")
        response = query_engine.query(query)
        # Print specific attributes of the response
        print(dir(response))
        # If you expect the response to be JSON, you can print it as follows:
        if isinstance(response, dict):
            for key, value in response.items():
                print(f"{key}: {value}")
        #print("Metadata:", response.metadata)
        #print("Response type:", type(response.response))
        #print("Response content:", response.response)
        #print("Source nodes:", response.source_nodes)
        formatted_response = {
        "content": response.response,  # main response content
        "metadata": response.metadata,  # include metadata if needed
        #"nodes": response.source_nodes  # include processed nodes
        }

        return formatted_response

    def query_llama_insight(self, query):
        query_engine = self._get_query_engine("LLamaInsightCode")
        response = query_engine.query(query)
        # Print specific attributes of the response
        print(dir(response))
        # If you expect the response to be JSON, you can print it as follows:
        if isinstance(response, dict):
            for key, value in response.items():
                print(f"{key}: {value}")
        #print("Metadata:", response.metadata)
        #print("Response type:", type(response.response))
        #print("Response content:", response.response)
        #print("Source nodes:", response.source_nodes)
        formatted_response = {
        "content": response.response,  # main response content
        "metadata": response.metadata,  # include metadata if needed
        #"nodes": response.source_nodes  # include processed nodes
        }

        return formatted_response

    def query_autogen(self, query):
        query_engine = self._get_query_engine("Autogen")
        response = query_engine.query(query)
        # Print specific attributes of the response
        print(dir(response))
        # If you expect the response to be JSON, you can print it as follows:
        if isinstance(response, dict):
            for key, value in response.items():
                print(f"{key}: {value}")
        #print("Metadata:", response.metadata)
        #print("Response type:", type(response.response))
        #print("Response content:", response.response)
        #print("Source nodes:", response.source_nodes)
        formatted_response = {
        "content": response.response,  # main response content
        "metadata": response.metadata,  # include metadata if needed
        #"nodes": response.source_nodes  # include processed nodes
        }

        return formatted_response

    def query_autogen_code(self, query):
        query_engine = self._get_query_engine("AutogenCode")
        response = query_engine.query(query)
        # Print specific attributes of the response
        print(dir(response))
        # If you expect the response to be JSON, you can print it as follows:
        if isinstance(response, dict):
            for key, value in response.items():
                print(f"{key}: {value}")
        #print("Metadata:", response.metadata)
        #print("Response type:", type(response.response))
        #print("Response content:", response.response)
        #print("Source nodes:", response.source_nodes)
        formatted_response = {
        "content": response.response,  # main response content
        "metadata": response.metadata,  # include metadata if needed
        #"nodes": response.source_nodes  # include processed nodes
        }

        return formatted_response

    def query_langchain(self, query):
        query_engine = self._get_query_engine("Langchain")
        response = query_engine.query(query)
        # Print specific attributes of the response
        print(dir(response))
        # If you expect the response to be JSON, you can print it as follows:
        if isinstance(response, dict):
            for key, value in response.items():
                print(f"{key}: {value}")
        #print("Metadata:", response.metadata)
        #print("Response type:", type(response.response))
        #print("Response content:", response.response)
        #print("Source nodes:", response.source_nodes)
        formatted_response = {
        "content": response.response,  # main response content
        "metadata": response.metadata,  # include metadata if needed
        #"nodes": response.source_nodes  # include processed nodes
        }

        return formatted_response

    def query_langchain_code(self, query):
        query_engine = self._get_query_engine("LangChainCode")
        response = query_engine.query(query)
        # Print specific attributes of the response
        print(dir(response))
        # If you expect the response to be JSON, you can print it as follows:
        if isinstance(response, dict):
            for key, value in response.items():
                print(f"{key}: {value}")
        #print("Metadata:", response.metadata)
        #print("Response type:", type(response.response))
        #print("Response content:", response.response)
        #print("Source nodes:", response.source_nodes)
        formatted_response = {
        "content": response.response,  # main response content
        "metadata": response.metadata,  # include metadata if needed
        #"nodes": response.source_nodes  # include processed nodes
        }

        return formatted_response

    def query_taskweaver(self, query):
        query_engine = self._get_query_engine("Taskweaver")
        response = query_engine.query(query)
        # Print specific attributes of the response
        print(dir(response))
        # If you expect the response to be JSON, you can print it as follows:
        if isinstance(response, dict):
            for key, value in response.items():
                print(f"{key}: {value}")
        #print("Metadata:", response.metadata)
        #print("Response type:", type(response.response))
        #print("Response content:", response.response)
        #print("Source nodes:", response.source_nodes)
        formatted_response = {
        "content": response.response,  # main response content
        "metadata": response.metadata,  # include metadata if needed
        #"nodes": response.source_nodes  # include processed nodes
        }

        return formatted_response

    def query_taskweaver_code(self, query):
        query_engine = self._get_query_engine("TaskWeaverCode")
        response = query_engine.query(query)
        # Print specific attributes of the response
        print(dir(response))
        # If you expect the response to be JSON, you can print it as follows:
        if isinstance(response, dict):
            for key, value in response.items():
                print(f"{key}: {value}")
        #print("Metadata:", response.metadata)
        #print("Response type:", type(response.response))
        #print("Response content:", response.response)
        #print("Source nodes:", response.source_nodes)
        formatted_response = {
        "content": response.response,  # main response content
        "metadata": response.metadata,  # include metadata if needed
        #"nodes": response.source_nodes  # include processed nodes
        }

        return formatted_response

    def query_embedchain(self, query):
        query_engine = self._get_query_engine("EmbedChain")
        response = query_engine.query(query)
        # Print specific attributes of the response
        print(dir(response))
        # If you expect the response to be JSON, you can print it as follows:
        if isinstance(response, dict):
            for key, value in response.items():
                print(f"{key}: {value}")
        #print("Metadata:", response.metadata)
        #print("Response type:", type(response.response))
        #print("Response content:", response.response)
        #print("Source nodes:", response.source_nodes)
        formatted_response = {
        "content": response.response,  # main response content
        "metadata": response.metadata,  # include metadata if needed
        #"nodes": response.source_nodes  # include processed nodes
        }

        return formatted_response

    def query_embedcode(self, query):
        query_engine = self._get_query_engine("EmbedCode")
        response = query_engine.query(query)
        # Print specific attributes of the response
        print(dir(response))
        # If you expect the response to be JSON, you can print it as follows:
        if isinstance(response, dict):
            for key, value in response.items():
                print(f"{key}: {value}")
        #print("Metadata:", response.metadata)
        #print("Response type:", type(response.response))
        #print("Response content:", response.response)
        #print("Source nodes:", response.source_nodes)
        formatted_response = {
        "content": response.response,  # main response content
        "metadata": response.metadata,  # include metadata if needed
        #"nodes": response.source_nodes  # include processed nodes
        }

        return formatted_response

    def query_nextjs(self, query):
        query_engine = self._get_query_engine("NextJS")
        response = query_engine.query(query)
        # Print specific attributes of the response
        print(dir(response))
        # If you expect the response to be JSON, you can print it as follows:
        if isinstance(response, dict):
            for key, value in response.items():
                print(f"{key}: {value}")
        #print("Metadata:", response.metadata)
        #print("Response type:", type(response.response))
        #print("Response content:", response.response)
        #print("Source nodes:", response.source_nodes)
        formatted_response = {
        "content": response.response,  # main response content
        "metadata": response.metadata,  # include metadata if needed
        #"nodes": response.source_nodes  # include processed nodes
        }

        return formatted_response

    def query_nextjs_code_real(self, query):
        query_engine = self._get_query_engine("NextJsCodeReal")
        response = query_engine.query(query)
        # Print specific attributes of the response
        print(dir(response))
        # If you expect the response to be JSON, you can print it as follows:
        if isinstance(response, dict):
            for key, value in response.items():
                print(f"{key}: {value}")
        #print("Metadata:", response.metadata)
        #print("Response type:", type(response.response))
        #print("Response content:", response.response)
        #print("Source nodes:", response.source_nodes)
        formatted_response = {
        "content": response.response,  # main response content
        "metadata": response.metadata,  # include metadata if needed
        #"nodes": response.source_nodes  # include processed nodes
        }

        return formatted_response

    def query_tailwindcss(self, query):
        query_engine = self._get_query_engine("TailWindCss")
        response = query_engine.query(query)
        # Print specific attributes of the response
        print(dir(response))
        # If you expect the response to be JSON, you can print it as follows:
        if isinstance(response, dict):
            for key, value in response.items():
                print(f"{key}: {value}")
        #print("Metadata:", response.metadata)
        #print("Response type:", type(response.response))
        #print("Response content:", response.response)
        #print("Source nodes:", response.source_nodes)
        formatted_response = {
        "content": response.response,  # main response content
        "metadata": response.metadata,  # include metadata if needed
        #"nodes": response.source_nodes  # include processed nodes
        }

        return formatted_response

    def query_tailwindcss_code(self, query):
        query_engine = self._get_query_engine("TailWindCssCode")
        response = query_engine.query(query)
        # Print specific attributes of the response
        print(dir(response))
        # If you expect the response to be JSON, you can print it as follows:
        if isinstance(response, dict):
            for key, value in response.items():
                print(f"{key}: {value}")
        #print("Metadata:", response.metadata)
        #print("Response type:", type(response.response))
        #print("Response content:", response.response)
        #print("Source nodes:", response.source_nodes)
        formatted_response = {
        "content": response.response,  # main response content
        "metadata": response.metadata,  # include metadata if needed
        #"nodes": response.source_nodes  # include processed nodes
        }

        return formatted_response

        
app = FastAPI()

# Modèle de requête
class QueryModel(BaseModel):
    query: str

# Initialisation du VectorStoreQueryEngine
llm = Anyscale(api_key="esecret_z844drr79tdnc5brvg4pnlfjz3",model="mistralai/Mixtral-8x7B-Instruct-v0.1",max_tokens=20048)
weaviate_url = "https://llamaindex-cluster-2tjlzjm6.weaviate.network"
api_key = "3j6hwSZoIY3YKewF7X0EFeKeEKrRV8RHUGTM"
service_context = ServiceContext.from_defaults(llm=llm)
query_engine = VectorStoreQueryEngine(weaviate_url, api_key, service_context)

# Utilisation du modèle de réponse dans votre route FastAPI
@app.post("/query_llama_lab/", response_model=QueryResponseModel)
async def query_llama_lab_endpoint(request: QueryModel = Body(...)):
    try:
        # Get the response text from the query_llama_lab method
        response_data = query_engine.query_llama_lab(request.query)
        return QueryResponseModel(success=True, data=response_data)
    except Exception as e:
        return QueryResponseModel(success=False, error=str(e))

# Ajoutez d'autres points de terminaison pour les autres fonctions de requête...

# Exemple pour query_llama_index
@app.post("/query_llama_index/")
async def query_llama_index(request: QueryModel = Body(...)):
    try:
        response_data = query_engine.query_llama_index(request.query)
         # Supposons que response_data est une liste de chaînes de caractères
        return QueryResponseModel(success=True, data=response_data)
    except Exception as e:
        return QueryResponseModel(success=False, error=str(e))
    
    # Utilisation du modèle de réponse dans votre route FastAPI
@app.post("/query_llama_hub/", response_model=QueryResponseModel)
async def query_llama_hub(request: QueryModel = Body(...)):
    try:
        # Get the response text from the query_llama_lab method
        response_data = query_engine.query_llama_hub(request.query)
        return QueryResponseModel(success=True, data=response_data)
    except Exception as e:
        return QueryResponseModel(success=False, error=str(e))

# Ajoutez d'autres points de terminaison pour les autres fonctions de requête...

# Exemple pour query_llama_index
@app.post("/query_llama_insight/")
async def query_llama_insight(request: QueryModel = Body(...)):
    try:
        response_data = query_engine.query_llama_insight(request.query)
         # Supposons que response_data est une liste de chaînes de caractères
        return QueryResponseModel(success=True, data=response_data)
    except Exception as e:
        return QueryResponseModel(success=False, error=str(e))

# Utilisation du modèle de réponse dans votre route FastAPI
@app.post("/query_autogen/", response_model=QueryResponseModel)
async def query_autogen(request: QueryModel = Body(...)):
    try:
        # Get the response text from the query_llama_lab method
        response_data = query_engine.query_autogen(request.query)
        return QueryResponseModel(success=True, data=response_data)
    except Exception as e:
        return QueryResponseModel(success=False, error=str(e))

# Ajoutez d'autres points de terminaison pour les autres fonctions de requête...

# Exemple pour query_llama_index
@app.post("/query_autogen_code/")
async def query_autogen_code(request: QueryModel = Body(...)):
    try:
        response_data = query_engine.query_autogen_code(request.query)
         # Supposons que response_data est une liste de chaînes de caractères
        return QueryResponseModel(success=True, data=response_data)
    except Exception as e:
        return QueryResponseModel(success=False, error=str(e))
    
    # Utilisation du modèle de réponse dans votre route FastAPI
@app.post("/query_langchain/", response_model=QueryResponseModel)
async def query_langchain(request: QueryModel = Body(...)):
    try:
        # Get the response text from the query_llama_lab method
        response_data = query_engine.query_langchain(request.query)
        return QueryResponseModel(success=True, data=response_data)
    except Exception as e:
        return QueryResponseModel(success=False, error=str(e))

# Ajoutez d'autres points de terminaison pour les autres fonctions de requête...

# Exemple pour query_llama_index
@app.post("/query_langchain_code/")
async def query_langchain_code(request: QueryModel = Body(...)):
    try:
        response_data = query_engine.query_langchain_code(request.query)
         # Supposons que response_data est une liste de chaînes de caractères
        return QueryResponseModel(success=True, data=response_data)
    except Exception as e:
        return QueryResponseModel(success=False, error=str(e))
# Exécuter avec uvicorn : uvicorn main:app --reload
# où 'main' est le nom de votre fichier Python.
    
@app.post("/query_taskweaver/", response_model=QueryResponseModel)
async def query_taskweaver(request: QueryModel = Body(...)):
    try:
        # Get the response text from the query_llama_lab method
        response_data = query_engine.query_taskweaver(request.query)
        return QueryResponseModel(success=True, data=response_data)
    except Exception as e:
        return QueryResponseModel(success=False, error=str(e))

# Ajoutez d'autres points de terminaison pour les autres fonctions de requête...

# Exemple pour query_llama_index
@app.post("/query_taskweaver_code/")
async def query_taskweaver_code(request: QueryModel = Body(...)):
    try:
        response_data = query_engine.query_taskweaver_code(request.query)
         # Supposons que response_data est une liste de chaînes de caractères
        return QueryResponseModel(success=True, data=response_data)
    except Exception as e:
        return QueryResponseModel(success=False, error=str(e))
    
    # Utilisation du modèle de réponse dans votre route FastAPI
@app.post("/query_embedchain/", response_model=QueryResponseModel)
async def query_embedchain(request: QueryModel = Body(...)):
    try:
        # Get the response text from the query_llama_lab method
        response_data = query_engine.query_embedchain(request.query)
        return QueryResponseModel(success=True, data=response_data)
    except Exception as e:
        return QueryResponseModel(success=False, error=str(e))

# Ajoutez d'autres points de terminaison pour les autres fonctions de requête...

# Exemple pour query_llama_index
@app.post("/query_embedcode/")
async def query_embedcode(request: QueryModel = Body(...)):
    try:
        response_data = query_engine.query_embedcode(request.query)
         # Supposons que response_data est une liste de chaînes de caractères
        return QueryResponseModel(success=True, data=response_data)
    except Exception as e:
        return QueryResponseModel(success=False, error=str(e))

# Utilisation du modèle de réponse dans votre route FastAPI
@app.post("/query_nextjs/", response_model=QueryResponseModel)
async def query_nextjs(request: QueryModel = Body(...)):
    try:
        # Get the response text from the query_llama_lab method
        response_data = query_engine.query_nextjs(request.query)
        return QueryResponseModel(success=True, data=response_data)
    except Exception as e:
        return QueryResponseModel(success=False, error=str(e))

# Ajoutez d'autres points de terminaison pour les autres fonctions de requête...

# Exemple pour query_llama_index
@app.post("/query_autogen_code/")
async def query_autogen_code(request: QueryModel = Body(...)):
    try:
        response_data = query_engine.query_autogen_code(request.query)
         # Supposons que response_data est une liste de chaînes de caractères
        return QueryResponseModel(success=True, data=response_data)
    except Exception as e:
        return QueryResponseModel(success=False, error=str(e))
    
# Utilisation du modèle de réponse dans votre route FastAPI
@app.post("/query_nextjs_code_real/", response_model=QueryResponseModel)
async def query_nextjs_code_real(request: QueryModel = Body(...)):
    try:
        # Get the response text from the query_llama_lab method
        response_data = query_engine.query_nextjs_code_real(request.query)
        return QueryResponseModel(success=True, data=response_data)
    except Exception as e:
        return QueryResponseModel(success=False, error=str(e))

# Ajoutez d'autres points de terminaison pour les autres fonctions de requête...

# Exemple pour query_llama_index
@app.post("/query_tailwindcss/")
async def query_tailwindcss(request: QueryModel = Body(...)):
    try:
        response_data = query_engine.query_tailwindcss(request.query)
         # Supposons que response_data est une liste de chaînes de caractères
        return QueryResponseModel(success=True, data=response_data)
    except Exception as e:
        return QueryResponseModel(success=False, error=str(e))

# Utilisation du modèle de réponse dans votre route FastAPI
@app.post("/query_nextjs_code_real/", response_model=QueryResponseModel)
async def query_nextjs_code_real(request: QueryModel = Body(...)):
    try:
        # Get the response text from the query_llama_lab method
        response_data = query_engine.query_nextjs_code_real(request.query)
        return QueryResponseModel(success=True, data=response_data)
    except Exception as e:
        return QueryResponseModel(success=False, error=str(e))

# Ajoutez d'autres points de terminaison pour les autres fonctions de requête...

# Exemple pour query_llama_index
@app.post("/query_tailwindcss_code/")
async def query_tailwindcss_code(request: QueryModel = Body(...)):
    try:
        response_data = query_engine.query_tailwindcss_code(request.query)
         # Supposons que response_data est une liste de chaînes de caractères
        return QueryResponseModel(success=True, data=response_data)
    except Exception as e:
        return QueryResponseModel(success=False, error=str(e))