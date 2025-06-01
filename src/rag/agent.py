from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict, Optional
from langchain_groq import ChatGroq
from langchain_together import ChatTogether
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.memory import ConversationBufferWindowMemory
from transformers import AutoTokenizer
from langdetect import detect, LangDetectException
import os
from langchain_community.tools import TavilySearchResults
from langchain_community.utilities import GooglePlacesAPIWrapper


from dotenv import load_dotenv
load_dotenv()

class ChatState(TypedDict):
    query: str
    language: str
    context: str
    history: str
    response: str
    intent: str
    error: str
    

class ConversationalRetrievalAgent:
    def __init__(self, indexer, temperature=0.2, max_history_tokens=1000, streaming=False):
        self.indexer = indexer
        self.temperature = temperature
        self.llm_response = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=temperature,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            streaming=True
        )
        self.llm_router = ChatGroq(
            model="gemma2-9b-it",
            temperature=0.1,
            max_tokens=None,
            max_retries=2
        )
        self.reranker = CrossEncoderReranker(
            model=HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        )
        
        self.max_history_tokens = max_history_tokens
        self.tokenizer = AutoTokenizer.from_pretrained( # for estimate token count
            "google/gemma-2-9b-it",
            token=os.getenv('HUGGINGFACE_API_KEY')
        )

        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=5,
            return_messages=False
        )
        
        self.search_tool = TavilySearchResults(
            max_results=3,
            topic="general",
            api_key=os.getenv('TAVILY_API_KEY')
        )

        self.gmaps = GooglePlacesAPIWrapper(top_k_results=5)

        self.tools = {
            "retrieve": self._retrieve,
            "search": self._web_search,
            "location": self._get_location_map
        }
        
                
        self.prompts = {
            "classify_intent": (
                "You are a routing assistant for a Nha Trang Tourism Chatbot. "
                "Given a user query and conversation history, determine the best action: 'retrieve' (fetch from pre-crawled documents), 'search' (web search for real-time data or weather data), or 'location' (get map/location of a place or a category of places). "
                "Consider the query's language and intent.\n"
                "Default to 'retrieve' for most travel-related queries (e.g., places to visit, activities, cuisine, culture, attractions) and generic/non-specific queries (e.g., greetings, casual conversation) to leverage the comprehensive Nha Trang travel database.\n"
                "Choose 'search' only for queries explicitly requesting real-time, recent information (e.g., containing 'today', 'now', 'current events', 'this week') or about weather conditions (e.g., 'weather', 'temperature', 'forecast', 'how it feel like', 'temperature').\n"
                "Choose 'location' for queries asking about the location, address, or map of a place, or for categories of places (such as restaurants, hotels, cafes, bars, spas, shops, etc.) in a specific area (e.g., 'Where is Vinpearl?', 'Show me the map of Po Nagar Tower', 'restaurants near District 1', 'quán ăn ngon tại Hòn Tằm', 'khách sạn gần biển', 'quán ăn ngon tại nha trang').\n"
                "Return only the action name: [retrieve|search|location]\n"
                "History: {history}\nLanguage: {language}"),
            "transform_query": (
                "You are a query optimization assistant for a keyword-based web search engine. Given a user query, rewrite it into a concise, keyword-focused search query optimized for keyword matching (e.g., like Google or Tavily). Remove unnecessary words, focus on key terms, and ensure clarity. Preserve the intent and language of the original query."
                "Return only the transformed query.\nQuery: {query}\nLanguage: {language}"),
            "generate_response": (
                "You are a travel chatbot for Nha Trang, Vietnam. Your role is to provide informative, concise, and accurate answers about travel-related topics (destinations, cuisine, festivals, activities, culture). "
                "Respond in the language specified by the user (ISO 639-1 code, e.g., 'vi' for Vietnamese, 'en' for English, 'zh' for Chinese). "
                "Use the provided context and conversation history to tailor your response. "
                "If the context contains a list of locations (with names and addresses), and the user asks for addresses, extract and list the addresses for each place in a clear, bullet-pointed format. "
                "If the user asks about a specific place, select the most relevant (usually the top) location from the context and respond with its name, address and the map url. "
                "If the user asks about places in general, summarize all the locations in the context, listing their names, addresses and map urls. "
                "Remember, all the locations are in Nha Trang, Vietnam. If user asks about another location outside Nha Trang, "
                "indicate that you only provide information about Nha Trang. "
                "If context is empty, rely on general knowledge or indicate a lack of specific information. "
                "Keep answers between 100 and 1000 words. Ensure responses are engaging, helpful, and culturally sensitive. "
                "History chat: {history}\nContext: {context}\nLanguage: {language}"),
            "translate_to_vi": (
                "Translate the following text to Vietnamese. Return only the translated text, without explanation or extra information.\n\nText: {text}"
            ),
            "resolve_place": (
                "Given the following conversation history and the latest user query, "
                "extract the most relevant place name or general place category the user wants to find the location for in Nha Trang. "
                "If the query refers to a previously mentioned place (e.g., 'this place', 'there'), "
                "replace it with the actual place name from the conversation. "
                "If the query is about a general category (e.g., 'restaurants near Hon Tam'), return that phrase. "
                "If you cannot determine any place or category, use the original query as the answer. "
                "Return only a single string: the most relevant place name or general place category. "
                "For example, if the query is about multiple places, return the general category (e.g., 'restaurants near Hon Tam'). "
                "Always add the location 'Nha Trang' to the place name or category. "
                "Return only the string, nothing else.\n\n"
                "Conversation history:\n{history}\n"
                "User query: {query}\n"
                "Place:"
            )
        }
        if streaming:
            self.app = self._build_streaming_graph()
        else:
            self.app = self._build_graph()
        

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))
    
    
    def _truncate_history(self, history: str) -> str:
        """Truncate history to fit within max_history_tokens."""
        
        if not history:
            return ""
        tokens = self._estimate_tokens(history)
        if tokens <= self.max_history_tokens:
            return history
        # Truncate by removing oldest interactions
        lines = history.split("\n")
        truncated = []
        total_tokens = 0
        for line in reversed(lines):
            line_tokens = self._estimate_tokens(line)
            if total_tokens + line_tokens > self.max_history_tokens:
                break
            truncated.insert(0, line)
            total_tokens += line_tokens
        return "\n".join(truncated)

    
    def detect_lang(self, state: ChatState) -> ChatState:
        """Detect query language"""
        if state["language"] and state["language"] != "":
            return state
        try:
            state["language"] = detect(state["query"])
        except LangDetectException:
            state["language"] = "vi" # Default to Vietnamese
        return state
    
    
    def _translate_to_vi(self, query) -> str:
        """Translate text to Vietnamese using LLM."""
        prompt = self.prompts["translate_to_vi"].format(text=query)
        messages = [("system", prompt), ("user", query)]
        try:
            response = self.llm_router.invoke(messages)
            return response.content.strip()
        except Exception as e:
            print(f"Translation failed: {str(e)}")
            return query  # Fallback to original text

    def _retrieve(self, query: str, language: str) -> Dict:
        """Retrieve documents using HybridIndexer and CrossEncoderReranker, with translation if needed."""
        try:
            query_vi = query
            if language != "vi":
                query_vi = self._translate_to_vi(query)

            docs = self.indexer.hybrid_search(query_vi, k=30, score_threshold=0.2)
            
            self.reranker.top_n = 10
            reranked = self.reranker.compress_documents(documents=docs, query=query_vi)
            context = "\n".join(d.page_content for d in reranked)
            return {"context":context, "error":""}
        except Exception as e:
            return {"context":"","error":str(e)}

    
    def _transform_query(self, query: str, language: str) -> str:
        """Transform query into keyword-optimized format using LLaMA 3 70B."""
        prompt = self.prompts["transform_query"].format(
            query=query,
            language=language
        )
        messages = [("system", prompt), ("user", query)]
        try:
            response = self.llm_router.invoke(messages)
            transformed_query = response.content.strip()
            return transformed_query if transformed_query else query
        except Exception as e:
            print(f"Query transformation failed: {str(e)}")
            return query  # Fallback to original query
    
    
    def _web_search(self, query: str, language: str) -> Dict:
        """Search web using Tavily api."""
        
        try:
            # Transform query for keyword-based search
            transformed_query = self._transform_query(query, language)
            results = self.search_tool.invoke(
                {
                    "query": transformed_query
                }
            )
            contents = "\n".join([r.get("content", "") for r in results])
            sources = [r.get("url", "") for r in results]
            context = f"Web search contents: {contents}\n\nSources: {sources}"
            return {
                "context": context,
                "error": "" if context else "No search results"
            }
        except Exception as e:
            return {"context": "", "error": str(e)}
    
    
    def _location_query(self, query: str, history:str) -> Dict:
        """Use LLM to rewrite the query for a location search, resolving references."""
        prompt = self.prompts["resolve_place"].format(history=history, query=query)
        messages = [("system", prompt), ("user", query)]
        try:
            response = self.llm_router.invoke(messages)
            search_query = response.content.strip()
            return search_query
        except Exception as e:
            print(f"Place resolution failed: {str(e)}")
            return query
    
    
    def _get_location_map(self, query: str, language: str) -> dict:
        """Get Google Maps info and links in Nha Trang"""
        try:
            history = self.memory.load_memory_variables({})["chat_history"]
            # Use LLM to extract a list of places/categories as a search query
            search_query = self._location_query(query, history)
            # Use GooglePlacesAPIWrapper.run (handles top_k_results internally)
            results = self.gmaps.run(search_query)
            map_url = f"https://www.google.com/maps/search/?api=1&query={search_query.replace(' ', '+')}"
            context = f"Locations: {results}\n Map: {map_url}"
            return {"context": context, "error": ""}
        except Exception as e:
            return {"context": "", "error": str(e)}
    
    
    def classify_intent(self, state: ChatState) -> ChatState:
        """Classify query intent using LLaMA 3 70B Instruct Turbo."""

        history = self._truncate_history(state["history"])
        prompt = self.prompts["classify_intent"].format(
            history=history, language=state["language"]
        )
        messages = [("system", prompt), ("user", state["query"])]
        try:
            response = self.llm_router.invoke(messages)
            intent = response.content
            state["intent"] = intent.strip()
            if state["intent"] not in self.tools:
                state["intent"] = "retrieve" if self.indexer.all_sections else "search"
        except Exception as e:
            state["intent"] = "retrieve" if self.indexer.all_sections else "search"
            state["error"] = f"Intent classification failed: {str(e)}"
        return state
    

    def execute_tool(self, state: ChatState) -> ChatState:
        """Execute the selected tool."""
        fn = self.tools.get(state["intent"])
        if not fn:
            state.update({"context":"","error":"Invalid intent"})
            return state
        print(f"Executing tool: {state['intent']}")
        res = fn(state["query"], state["language"])
        state.update(res)
        return state
            
            
    def handle_error(self, state: ChatState) -> ChatState:
        """Handle error from tool excution"""
        
        if not state["error"] or state["intent"]=="search":
            return state
        # fallback to search
        try:
            print(f"Error occurred: {state['error']}. Fallback to web search.")
            res = self._web_search(state["query"], state["language"])
            state.update({"intent":"search", **res, "error":""})
        except Exception as e:
            state["error"] = f"Fallback search failed: {e}"
        return state

    
    
    def generate_response(self, state: ChatState) -> ChatState:
        """Generate response"""
        
        hist = self._truncate_history(state["history"])
        prompt = self.prompts["generate_response"].format(
            history=hist, context=state["context"], language=state["language"]
        )
        messages = [("system", prompt), ("user", state["query"])]
        try:
            response = self.llm_response.invoke(messages)
            out = response.content
            state["response"] = out
        except Exception as e:
            msg = ("Sorry, error generating response." if state["language"]=="en"
                   else "Xin lỗi, không thể tạo câu trả lời.")
            state["response"] = f"{msg} (Error: {e})"
            state["error"] = str(e)
        # save memory
        self.memory.save_context(inputs={"human":state["query"]}, outputs={"ai":state["response"]})
        print(state)
        return state
    
    
    def _build_graph(self):
        """Build LangGraph workflow using ToolNode for tool execution."""
        graph = StateGraph(ChatState)
        graph.add_node("detect_language", self.detect_lang)
        graph.add_node("classify_intent", self.classify_intent)
        # Use ToolNode for tool execution
        graph.add_node("execute_tool", self.execute_tool)
        graph.add_node("handle_error", self.handle_error)
        graph.add_node("generate_response", self.generate_response)

        graph.set_entry_point("detect_language")
        graph.add_edge("detect_language", "classify_intent")
        graph.add_edge("classify_intent", "execute_tool")
        graph.add_edge("execute_tool", "handle_error")
        graph.add_conditional_edges(
            "handle_error",
            lambda s: "generate_response" if not s["error"] or s["intent"] == "search" else "execute_tool",
            {"generate_response": "generate_response", "execute_tool": "execute_tool"}
        )
        graph.add_edge("generate_response", END)
        return graph.compile()
    
    
    def _build_streaming_graph(self):
        """Build LangGraph workflow for streaming responses."""
        graph = StateGraph(ChatState)
        graph.add_node("detect_language", self.detect_lang)
        graph.add_node("classify_intent", self.classify_intent)
        graph.add_node("execute_tool", self.execute_tool)
        graph.add_node("handle_error", self.handle_error)

        graph.set_entry_point("detect_language")
        graph.add_edge("detect_language", "classify_intent")
        graph.add_edge("classify_intent", "execute_tool")
        graph.add_edge("execute_tool", "handle_error")
        graph.add_edge("handle_error", END)  # End here, response generation is handled separately
        
        return graph.compile()
    
    
    def stream_response(self, query, language, context, history):
        """
        Generate and yield response chunks using the language model's streaming capability.
        """
        hist = self._truncate_history(history)
        prompt = self.prompts["generate_response"].format(
            history=hist, context=context, language=language
        )
        messages = [("system", prompt), ("user", query)]
        try:
            # Stream response chunks from llm_response (ChatGroq with streaming=True)
            for chunk in self.llm_response.stream(messages):
                text = chunk.content if hasattr(chunk, "content") else str(chunk)
                yield text  # Yield the content of each chunk
        except Exception as e:
            yield f"Error: {str(e)}"
    
    
    def ask_streaming(self, question, language=None):
        """
        Ask a question and return a generator that yields response chunks.
        This is useful for streaming responses in real-time.
        """
        history = self.memory.load_memory_variables({})["chat_history"]
        init_state = {
            "query": question,
            "language": language if language else "",
            "context": "",
            "history": history,
            "intent": "",
            "error": ""
        }
        # Run the workflow up to error handling
        final = self.app.invoke(init_state)
        full_response = ""
        
        def generate():
            nonlocal full_response
            # Stream response chunks
            for chunk in self.stream_response(final["query"], final["language"], final["context"], final["history"]):
                full_response += chunk
                yield chunk
            # Save the full response to memory after streaming
            self.memory.save_context(inputs={"human": question}, outputs={"ai": full_response})
        
        return generate()
        
    
    
    def ask(self, question, language=None):
        history = self.memory.load_memory_variables({})["chat_history"]  # already formatted string
        init_state: ChatState = {
            "query": question,
            "language": language if language else "",
            "context": "",
            "history": history,
            "response": "",
            "intent": "",
            "error": ""
        }
        final = self.app.invoke(init_state)
        
        return final["response"]
    
    
    
    