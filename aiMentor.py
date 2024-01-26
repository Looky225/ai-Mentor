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
            similarity_top_k=6
        )
        return query_engine

    def query_llama_index(self, query):
        query_engine = self._get_query_engine("LlamaIndex")
        response = query_engine.query(query)
        response.print_response_stream()

    def query_llama_hub(self, query):
        query_engine = self._get_query_engine("LLamaHubCode")
        response = query_engine.query(query)
        response.print_response_stream()

    def query_llama_lab(self, query):
        query_engine = self._get_query_engine("LLamaLabCode")
        response = query_engine.query(query)
        response.print_response_stream()

    def query_llama_insight(self, query):
        query_engine = self._get_query_engine("LLamaInsightCode")
        response = query_engine.query(query)
        response.print_response_stream()

    def query_autogen(self, query):
        query_engine = self._get_query_engine("Autogen")
        response = query_engine.query(query)
        response.print_response_stream()

    def query_autogen_code(self, query):
        query_engine = self._get_query_engine("AutogenCode")
        response = query_engine.query(query)
        response.print_response_stream()

    def query_langchain(self, query):
        query_engine = self._get_query_engine("Langchain")
        response = query_engine.query(query)
        response.print_response_stream()

    def query_langchain_code(self, query):
        query_engine = self._get_query_engine("LangChainCode")
        response = query_engine.query(query)
        response.print_response_stream()

    def query_taskweaver(self, query):
        query_engine = self._get_query_engine("Taskweaver")
        response = query_engine.query(query)
        response.print_response_stream()

    def query_taskweaver_code(self, query):
        query_engine = self._get_query_engine("TaskWeaverCode")
        response = query_engine.query(query)
        response.print_response_stream()

    def query_embedchain(self, query):
        query_engine = self._get_query_engine("EmbedChain")
        response = query_engine.query(query)
        response.print_response_stream()

    def query_embedcode(self, query):
        query_engine = self._get_query_engine("EmbedCode")
        response = query_engine.query(query)
        response.print_response_stream()

    def query_nextjs(self, query):
        query_engine = self._get_query_engine("NextJS")
        response = query_engine.query(query)
        return str(response)

    def query_nextjs_code_real(self, query):
        query_engine = self._get_query_engine("NextJsCodeReal")
        response = query_engine.query(query)
        return str(response)

    def query_tailwindcss(self, query):
        query_engine = self._get_query_engine("TailWindCss")
        response = query_engine.query(query)
        response.print_response_stream()

    def query_tailwindcss_code(self, query):
        query_engine = self._get_query_engine("TailWindCssCode")
        response = query_engine.query(query)
        response.print_response_stream()

    # Ajoutez d'autres méthodes pour les autres indexes...


llm = Anyscale(api_key="esecret_z844drr79tdnc5brvg4pnlfjz3",model="mistralai/Mixtral-8x7B-Instruct-v0.1",max_tokens=20548)

weaviate_url = "https://llamaindex-cluster-2tjlzjm6.weaviate.network"
api_key = "3j6hwSZoIY3YKewF7X0EFeKeEKrRV8RHUGTM"
service_context = ServiceContext.from_defaults(llm=llm)

query_engine = VectorStoreQueryEngine(weaviate_url, api_key, service_context)



# Interrogation de NextJS
response = query_engine.query_nextjs("Why does components needs to be imported why not have them in the same files ?")

#print(str(response))

do = str(response)

print(do)
"""
# Interrogation de LLamaLabCode
query_engine.query_llama_lab("show me the code for conversational agents")

# Interrogation de LlamaIndex
query_engine.query_llama_index("What is LlamaIndex and how does it work ?")

# Interrogation de LLamaHubCode
query_engine.query_llama_hub("How do i use the github loader ?")

# Interrogation de LLamaInsightCode
query_engine.query_llama_insight("Votre requête pour LLamaInsightCode")

# Interrogation de Autogen
query_engine.query_autogen("Votre requête pour Autogen")

# Interrogation de AutogenCode
query_engine.query_autogen_code("Votre requête pour AutogenCode")

# Interrogation de Langchain
query_engine.query_langchain("Votre requête pour Langchain")

# Interrogation de LangChainCode
query_engine.query_langchain_code("Votre requête pour LangChainCode")

# Interrogation de Taskweaver
query_engine.query_taskweaver("Votre requête pour Taskweaver")

# Interrogation de TaskweaverCode
query_engine.query_taskweaver_code("Votre requête pour TaskWeaverCode")

# Interrogation de EmbedChain
query_engine.query_embedchain("Votre requête pour EmbedChain")

# Interrogation de EmbedCode
query_engine.query_embedcode("Votre requête pour EmbedCode")



# Interrogation de NextJsCodeReal
query_engine.query_nextjs_code_real("Votre requête pour NextJsCodeReal")

# Interrogation de TailWindCss
query_engine.query_tailwindcss("Votre requête pour TailWindCss")

# Interrogation de TailWindCssCode
query_engine.query_tailwindcss_code("Votre requête pour TailWindCssCode")
"""


