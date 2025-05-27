from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict
from langchain_groq import ChatGroq
from together import Together
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.memory import ConversationBufferWindowMemory
from transformers import AutoTokenizer
from langdetect import detect, LangDetectException
import os
from langchain_community.tools import TavilySearchResults
from langchain_community.utilities import OpenWeatherMapAPIWrapper

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
    def __init__(self, indexer, temperature=0.5, max_history_tokens=1000):
        self.indexer = indexer
        self.temperature = temperature
        self.llm_response = ChatGroq(
            api_key=os.getenv('GROQ_API_KEY'),
            model="gemma2-9b-it",
            temperature=temperature,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            streaming=True
        )
        self.llm_router = Together(api_key=os.getenv('TOGETHER_API_KEY'))
        self.reranker = CrossEncoderReranker(
            model=HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        )
        
        self.max_history_tokens = max_history_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
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
        
        self.weather = OpenWeatherMapAPIWrapper(
            openweathermap_api_key=os.getenv("OPEN_WEATHER_API_KEY")
        )
        
        self.tools = {
            "retrieve": self._retrieve,
            "search": self._web_search,
            "weather": lambda q, lang: self._weather()
        }
        
        self.prompts = {
            "classify_intent": (
                "You are a routing assistant for a Nha Trang Tourism Chatbot."
                "Given a user query and conversation history, determine the best action: 'retrieve' (fetch from pre-crawled documents), 'search' (web search for real-time data), or 'weather' (fetch weather data)."
                "Consider the query's language and intent.\n"
                "Default to 'retrieve' for most travel-related queries (e.g., places to visit, activities, cuisine, culture, attractions), as the database contains comprehensive Nha Trang travel information.\n"
                "Choose 'search' only for queries explicitly requesting real-time or recent information (e.g., containing 'today', 'now', 'current events', 'this week').\n"
                "Choose 'weather' for queries about weather conditions (e.g., 'weather', 'temperature', 'forecast', 'how it feel like', 'temperature').\n"
                "Return only the action name: [retrieve|search|weather]\n"
                "History: {history}\nQuery: {query}\nLanguage: {language}"),
            "transform_query": (
                "You are a query optimization assistant for a keyword-based web search engine. Given a user query, rewrite it into a concise, keyword-focused search query optimized for keyword matching (e.g., like Google or Tavily). Remove unnecessary words, focus on key terms, and ensure clarity. Preserve the intent and language of the original query."
                "Return only the transformed query.\nQuery: {query}\nLanguage: {language}"),
            "generate_response": (
                "You are a travel chatbot for Nha Trang, Vietnam. Your role is to provide informative, concise, and accurate answers about travel-related topics (destinations, cuisine, festivals, activities, culture). Respond in the language specified by the user (ISO 639-1 code, e.g., 'vi' for Vietnamese, 'en' for English, 'zh' for Chinese). Use the provided context and conversation history to tailor your response. If context is empty, rely on general knowledge or indicate a lack of specific information. Keep answers between 100 and 700 words. Ensure responses are engaging, helpful, and culturally sensitive."
                "History chat: {history}\nContext: {context}\nLanguage: {language}")
        }
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
        if state["language"]:
            return state
        try:
            state["language"] = detect(state["query"])
        except LangDetectException:
            state["language"] = "vi" # Default to Vietnamese
        return state
    
    
    def _retrieve(self, query: str, language: str) -> Dict:
        """Retrieve documents using HybridIndexer and CrossEncoderReranker."""
        try:
            docs = self.indexer.hybrid_search(query, k=30, score_threshold=0.0)
            if not docs:
                return {"context":"","error":"No documents found"}
            self.reranker.top_n = 10
            reranked = self.reranker.compress_documents(documents=docs, query=query)
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
        try:
            response = self.llm_router.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=70
            )
            transformed_query = response.choices[0].message.content.strip()
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
    
    
    def _weather(self) -> Dict:
        """Fetch Nha Trang weather data with OpenWeatherMap."""
        
        try:
            weather_data = self.weather.run("NhaTrang, VN")
            context = f"Weather info: {weather_data}"
            return {"context": context ,"error": "" if context else "No weather data"}
        except Exception as e:
            return {"context": "", "error": str(e)}
    
    
    def classify_intent(self, state: ChatState) -> ChatState:
        """Classify query intent using LLaMA 3 70B Instruct Turbo."""
        history = self._truncate_history(state["history"])
        prompt = self.prompts["classify_intent"].format(
            history=history,
            query=state["query"],
            language=state["language"]
        )
        try:
            response = self.llm_router.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=10
            )
            state["intent"] = response.choices[0].message.content.strip()
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
        res = fn(state["query"], state["language"])  # weather ignores args
        state.update(res)
        return state
            
            
    def handle_error(self, state: ChatState) -> ChatState:
        """Handle error from tool excution"""
        if not state["error"] or state["intent"]=="search":
            return state
        # fallback to search
        try:
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
        return state
    
    
    def _build_graph(self):
        """Build LangGraph workflow."""
        graph = StateGraph(ChatState)
        graph.add_node("detect_language", self.detect_lang)
        graph.add_node("classify_intent", self.classify_intent)
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
    
    
    
    def ask(self, question, language=None):
        history = self.memory.load_memory_variables({})["chat_history"]  # already formatted string
        init_state: ChatState = {
            "query": question,
            "language": language or "",
            "context": "",
            "history": history,
            "response": "",
            "intent": "",
            "error": ""
        }
        final = self.app.invoke(init_state)
        self.memory.save_context(inputs={"human": question}, outputs={"ai": final["response"]})
        return final["response"]