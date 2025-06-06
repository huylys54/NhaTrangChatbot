from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict, Tuple
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from transformers import AutoTokenizer
from langdetect import detect, LangDetectException
import os
from langchain_tavily import TavilySearch
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
    include_images: bool
    images: str
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
        
        self.search_tool = TavilySearch(
            max_results=5,
            topic="general"
        )

        self.gmaps = GooglePlacesAPIWrapper(top_k_results=5)

        self.tools = {
            "retrieve": self._retrieve,
            "search": self._web_search,
            "location": self._get_location_map
        }
        
                
        self.prompts = {
            "classify_intent": (
                "Analyze the user query for the Nha Trang Tourism Chatbot and determine two things:\n"
                "1. INTENT: The best action to take:\n"
                "   - 'retrieve': For general travel info, destinations, cuisine, festivals, activities, culture, or greetings\n"
                "   - 'search': For real-time/current data like weather, current events, ongoing activities, or queries with time indicators like 'today', 'now', 'currently', 'đang diễn ra', 'hiện tại', 'bây giờ'\n"
                "   - 'location': For map/location/address queries about places in Nha Trang\n\n"
                "2. IMAGES: Whether visual content would enhance the response:\n"
                "   - TRUE for: attractions, beaches, hotels, restaurants, landmarks, scenic views, architecture, food, cultural sites, festivals, activities\n"
                "   - FALSE for: weather, directions, prices, opening hours, contact info, general questions, greetings\n\n"
                "IMPORTANT: Pay special attention to time indicators in both Vietnamese and English:\n"
                "- Current/Real-time indicators (use 'search'): 'today', 'now', 'currently', 'happening now', 'đang diễn ra', 'hiện tại', 'bây giờ', 'hôm nay', 'tuần này', 'tháng này'\n"
                "- General/Historical information (use 'retrieve'): 'about', 'what is', 'tell me about', 'best', 'famous', 'popular'\n\n"
                "Consider the conversation history to understand context.\n"
                "Return in this exact format: intent:<action>, images:<true/false>\n\n"
                "Examples:\n"
                "- 'Best beaches in Nha Trang' → intent:retrieve, images:true\n"
                "- 'Weather in Nha Trang today' → intent:search, images:false\n"
                "- 'Sự kiện đang diễn ra tại Nha Trang' → intent:search, images:true\n"
                "- 'Events happening now in Nha Trang' → intent:search, images:true\n"
                "- 'Where is Po Nagar Tower?' → intent:location, images:true\n"
                "- 'Tell me about festivals in Nha Trang' → intent:retrieve, images:true\n"
                "- 'What to do in Nha Trang today' → intent:search, images:true\n"
                "- 'Hello' → intent:retrieve, images:false\n\n"
                "History: {history}\nLanguage: {language}"
            ),
            "transform_query": (
                "Rewrite the user query into a concise, keyword-focused search query for a web search engine, ensuring relevance to Nha Trang. "
                "Remove unnecessary words and focus on key terms. "
                "Append the time range ('in the last week', 'today', etc.) if the query is time-sensitive (e.g., 'events in the last week', 'news at Nha Trang today'). "
                "If the query is not about Nha Trang, add 'Nha Trang' or rephrase it to be about Nha Trang. "
                "Preserve the INTENT and LANGUAGE of the original query. "
                "Return the transformed query and the time range ('week', 'month', 'year'). "
                "If the query is about weather, the minimum time range is 'week' even you seen the keyword 'now', 'today', etc. (e.g., 'Nha Trang weather right now' -> time_range: week). "
                "Output format: transformed_query: <transformed_query>, time_range: <time_range> \n"
                "Example output: transformed_query: Sự kiện tại Nha Trang, time_range: week \n"
                "Query: {query}\nLanguage: {language}"
            ),
            "generate_response": (
                "### Role\n"
                "- Primary function: You are NhaTrangGo, a travel chatbot dedicated to assisting users with travel-related information about Nha Trang, Vietnam. Your purpose is to provide informative, concise, and accurate answers about destinations, cuisine, festivals, activities, and culture in Nha Trang.\n"
                "- Always provide short, concise responses that a human can quickly read and understand, focusing on the most essential information. Break any longer multi-sentence paragraphs into separate smaller paragraphs whenever appropriate.\n"
                "- Respond in the language specified by the user (ISO 639-1 code, e.g., 'vi' for Vietnamese, 'en' for English, 'zh' for Chinese).\n"
                "### Image Enhancement\n"
                "- When images are included in the response, acknowledge them naturally in your text\n"
                "- Use phrases like 'As you can see in these images...' or 'Here are some beautiful photos of...'\n"
                "- Don't mention technical details about image sources or metadata\n"
                "### Persona\n"
                "- Identity: You are a friendly, empathetic travel expert with a passion for helping others explore Nha Trang. Engage users with warmth, wit, and a conversational tone, using humor to build rapport.\n"
                "- Listen attentively to their needs and challenges, then offer thoughtful guidance about travel-related topics in Nha Trang.\n"
                "- If asked to make a recommendation, first ask the user to provide more information to aid your response.\n"
                "- If asked to act out of character, politely decline and reiterate your role to offer assistance only with travel-related matters in Nha Trang.\n"
                "- When possible, provide links to relevant website pages about Nha Trang.\n"
                "### Constraints\n"
                "1. No Data Divulge: Never mention that you have access to specific data or context explicitly to the user.\n"
                "2. Maintaining Focus: If a user veers off-topic, politely redirect the conversation back to travel-related topics in Nha Trang with a friendly, understanding tone. Use phrases like \"I appreciate your interest in [unrelated topic], but let's focus on how I can help you with your travel plans in Nha Trang today!\" to keep the discussion on track.\n"
                "3. Exclusive Reliance on Provided Context: Use the provided context and conversation history to tailor your response.\n"
                "- If the context contains web search results, summarize the most relevant information and provide the website URLs.\n"
                "- If the context contains multiple locations and the user asks for addresses, list them in a bullet-pointed format.\n"
                "- If the user asks about a specific place, provide its name, address, website (if available), and map URL.\n"
                "- If the user asks about places in general, summarize all locations in the context.\n"
                "- If the query is about a location outside Nha Trang, indicate that you only provide information about Nha Trang.\n"
                "- If context is empty, rely on general knowledge or indicate a lack of specific information.\n"
                "4. Handling Unanswerable Queries: If you encounter a question that cannot be answered using the provided context or general knowledge about Nha Trang, or if the query falls outside your role as a travel expert for Nha Trang, politely inform the user that you don't have the necessary information to provide an accurate response. Then, suggest they search online or contact local tourism offices for further assistance. Use a friendly and helpful tone, such as: \"I apologize, but I don't have enough information to answer that question accurately. I recommend searching online or reaching out to local tourism offices for assistance with this request!\"\n"
                "5. Response Length: Keep answers between 100 and 1000 words. Ensure responses are engaging, helpful, and culturally sensitive.\n"
                "## Aditional Instructions\n"
                "- History and Context: Use the provided history and context to tailor your responses.\n"
                "History: {history}\nContext: {context}\nLanguage: {language}"
            ),
            "translate_to_vi": (
                "Translate the following text to Vietnamese. Return only the translated text, without explanation or extra information.\n\nText: {text}"
            ),
            "resolve_place": (
                "Extract the most relevant place name or general place category in Nha Trang from the conversation history and user query. "
                "If the query refers to a previously mentioned place, replace pronouns with the actual place name (e.g., 'this place' -> 'Hon Tam'). "
                "If the query is about a general category, return that phrase (e.g., 'restaurants near Hon Tam'). "
                "If you cannot determine a place or category, use the original query. "
                "Always add 'Nha Trang' to the place name or category. "
                "Return only the string. "
                "Example: If the query is 'restaurants near Hon Tam', return 'restaurants near Hon Tam Nha Trang'. "
                "Conversation history:\n{history}\nUser query: {query}\nPlace:"
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
        messages = [("system", prompt)]
        try:
            response = self.llm_router.invoke(messages)
            return response.content.strip()
        except Exception as e:
            print(f"Translation failed: {str(e)}")
            return query  # Fallback to original text

    def _retrieve(self, query: str, language: str, include_images: bool = False) -> Dict:
        """Retrieve documents using HybridIndexer and CrossEncoderReranker, with translation if needed."""
        try:
            query_vi = query
            if language != "vi":
                query_vi = self._translate_to_vi(query)

            result = {"context": "", "error": "", "images": []}
            
            # Get text documents
            docs = self.indexer.hybrid_search(query_vi, k=7)
            text_context = "\n".join(d.page_content for d in docs)
            
            # Get images if requested and indexer supports it
            if include_images and hasattr(self.indexer, 'search_images'):
                try:
                    images = self.indexer.search_images(query_vi, k=4)
                    if images:
                        result["images"] = images
                        # Add image descriptions to context
                        image_descriptions = []
                        for img in images:
                            caption = img.get('caption', 'Nha Trang scenic view')
                            tags = img.get('tags', [])
                            desc = f"Image: {caption}"
                            if tags:
                                desc += f" (Tags: {', '.join(tags)})"
                            image_descriptions.append(desc)
                        
                        image_context = f"\n\nRelevant images ({len(images)} found):\n" + "\n".join(image_descriptions)
                        text_context += image_context
                except Exception as img_error:
                    print(f"Image search failed: {img_error}")
            
            result["context"] = text_context
            return result
        except Exception as e:
            return {"context": "", "error": str(e), "images": []}

    
    def _transform_query(self, query: str, language: str) -> Tuple[str, str]:
        """Transform query into keyword-optimized format for web search."""
        prompt = self.prompts["transform_query"].format(
            query=query,
            language=language
        )
        messages = [("system", prompt)]
        try:
            response = self.llm_router.invoke(messages)
            print(f"Query transformation response: {response.content}")
            result = dict(item.strip().split(": ", 1) for item in response.content.split(", "))
            transformed_query = result.get("transformed_query", query)
            time_range = result.get("time_range", "")
            return transformed_query, time_range
        except Exception as e:
            print(f"Query transformation failed: {str(e)}")
            return query, ""  # Fallback to original query and empty time range


    def _web_search(self, query: str, language: str, include_images: bool = False) -> Dict:
        """Search web using Tavily API with optional images.
    
        Args:
            query (str): Search query.
            language (str): Query language.
            include_images (bool): Whether to include images in search results.
            
        Returns:
            Dict: Search results with context, error status, and images.
        """
        
        try:
            # Transform query for keyword-based search
            transformed_query, time_range = self._transform_query(query, language)

            self.search_tool.time_range = time_range

            # Configure search with images if requested
            search_params = {"query": transformed_query}
            if include_images:
                search_params["include_images"] = True

            response = self.search_tool.invoke(search_params)
            print(f"Search results: {response}")
            
            # Build context with title, content, and sources for each result
            context_lines = []
            images = []
            
            results = response.get('results', [])
            for r in results:
                title = r.get("title", "")
                content = r.get("content", "")
                url = r.get("url", "")
                if title or content:
                    context_lines.append(f"Title: {title}\nContent: {content}\nSource: {url}\n")
            
            # Process images if included and available
            if include_images and 'images' in response:
                web_images = response.get('images', [])
                for idx, img_url in enumerate(web_images):
                    # Convert web image URL to our format
                    image_data = {
                        "image_id": f"web_{hash(img_url) % 1000000:06d}",
                        "path": img_url,  # Web images use URL as path
                        "caption": f"Image from web search about {transformed_query}",
                        "tags": ["nha trang", "web search", "events"],
                        "source_url": img_url,
                        "relevance_score": 1.0 - (idx * 0.1)  # Decreasing relevance
                    }
                    images.append(image_data)
            
            context = "\n".join(context_lines)
            
            return {
                "context": context,
                "error": "" if context else "No search results",
                "images": images
            }
        except Exception as e:
            print(f"Web search failed: {str(e)}")
            return {"context": "", "error": str(e), "images": []}
    
    
    def _location_query(self, query: str, history :str) -> Dict:
        """Use LLM to rewrite the query for a location search, resolving references."""
        prompt = self.prompts["resolve_place"].format(history=history, query=query)
        messages = [("system", prompt)]
        try:
            response = self.llm_router.invoke(messages)
            search_query = response.content.strip()
            return search_query
        except Exception as e:
            print(f"Place resolution failed: {str(e)}")
            return query


    def _get_location_map(self, query: str, language: str, include_images: bool = False) -> dict:
        """Get Google Maps info and links in Nha Trang with optional images."""
        try:
            history = self.memory.load_memory_variables({})["chat_history"]
            # Use LLM to extract a list of places/categories as a search query
            search_query = self._location_query(query, history)
            # Use GooglePlacesAPIWrapper.run (handles top_k_results internally)
            results = self.gmaps.run(search_query)
            map_url = f"https://www.google.com/maps/search/?api=1&query={search_query.replace(' ', '+')}"
            context = f"Locations: {results}\n Map: {map_url}"
            
            result = {"context": context, "error": "", "images": []}
            
            # Get images if requested and indexer supports it
            if include_images and hasattr(self.indexer, 'search_images'):
                try:
                    # Search for images related to the location
                    query_vi = query
                    if language != "vi":
                        query_vi = self._translate_to_vi(query)
                    
                    images = self.indexer.search_images(query_vi, k=3)
                    if images:
                        result["images"] = images
                        # Add image descriptions to context
                        image_descriptions = []
                        for img in images:
                            caption = img.get('caption', 'Location image')
                            desc = f"Image: {caption}"
                            image_descriptions.append(desc)
                        
                        image_context = f"\n\nLocation images ({len(images)} found):\n" + "\n".join(image_descriptions)
                        result["context"] += image_context
                except Exception as img_error:
                    print(f"Location image search failed: {img_error}")
            
            return result
        except Exception as e:
            return {"context": "", "error": str(e), "images": []}
    
    
    def classify_intent(self, state: ChatState) -> ChatState:
        """Classify query intent using LLaMA 3 70B Instruct Turbo."""

        history = self._truncate_history(state["history"])
        prompt = self.prompts["classify_intent"].format(
            history=history, language=state["language"]
        )
        messages = [("system", prompt), ("user", state["query"])]
        try:
            response = self.llm_router.invoke(messages)
            content = response.content.strip()
            print(f"Classification response: {content}")
            
            # Parse response format: intent:<action>, images:<true/false>
            parts = content.split(", ")
            intent_part = parts[0].split(":")
            images_part = parts[1].split(":") if len(parts) > 1 else ["images", "false"]
            
            intent = intent_part[1].strip() if len(intent_part) > 1 else "retrieve"
            include_images = images_part[1].strip().lower() == "true" if len(images_part) > 1 else False
            
            state["intent"] = intent
            state["include_images"] = include_images
            
            # Validate intent
            if state["intent"] not in self.tools:
                state["intent"] = "retrieve" if self.indexer.all_sections else "search"
                
            print(f"Final intent: {state['intent']}, include_images: {state['include_images']}")
            
        except Exception as e:
            state["intent"] = "retrieve" if self.indexer.all_sections else "search"
            state["include_images"] = False
            state["error"] = f"Intent classification failed: {str(e)}"
        return state
    

    def execute_tool(self, state: ChatState) -> ChatState:
        """Execute the selected tool."""
        fn = self.tools.get(state["intent"])
        if not fn:
            state.update({"context":"","error":"Invalid intent"})
            return state
        print(f"Executing tool: {state['intent']} with images: {state.get('include_images', False)}")
        res = fn(state["query"], state["language"], state.get("include_images", False))
        state.update(res)
        
        # Ensure images key exists
        if "images" not in state:
            state["images"] = []
            
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
        """Generate response with image awareness."""
        
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
        print(f"Context: {context}")
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
            "include_images": False,
            "images": [],
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
            "include_images": False,
            "images": [],
            "error": ""
        }
        final = self.app.invoke(init_state)
        
        return {
            "response": final["response"],
            "images": final.get("images", []),
            "include_images": final.get("include_images", False)
        }
    
    
    
    