import streamlit as st
import requests
import json
from PIL import Image
from pathlib import Path
import io

# Page config
st.set_page_config(
    page_title="NhaTrangGo",
    page_icon="🏖️",
    layout="wide"
)

# API configuration
API_BASE_URL = "http://localhost:8000"

def display_images(images_data):
    """Display images in a grid layout.
    
    Args:
        images_data (list): List of image data dictionaries.
    """
    if not images_data:
        return
    
    # Filter out images that can't be loaded and collect valid ones
    valid_images = []
    
    for img_data in images_data:
        try:
            img_path = img_data["path"]
            
            # Check if it's a web URL or local file
            if img_path.startswith(('http://', 'https://')):
                # Handle web images - test if accessible
                try:
                    response = requests.get(img_path, timeout=5, stream=True)
                    response.raise_for_status()
                    
                    # Check content type
                    content_type = response.headers.get('content-type', '').lower()
                    if content_type.startswith('image/'):
                        valid_images.append(img_data)
                    # Silently skip if not an image
                except:
                    # Silently skip unavailable web images
                    continue
            else:
                # Handle local files
                local_path = Path(img_path)
                if local_path.exists():
                    valid_images.append(img_data)
                # Silently skip missing local files
        except:
            # Silently skip any problematic image data
            continue
    
    # Only show images section if we have valid images
    if not valid_images:
        return
    
    st.subheader("📸 Related Images")
    
    # Create columns for image grid
    cols_per_row = 3
    rows = [valid_images[i:i + cols_per_row] for i in range(0, len(valid_images), cols_per_row)]
    
    for row in rows:
        cols = st.columns(len(row))
        for col, img_data in zip(cols, row):
            with col:
                try:
                    img_path = img_data["path"]
                    
                    if img_path.startswith(('http://', 'https://')):
                        # Load web image
                        response = requests.get(img_path, timeout=10)
                        image = Image.open(io.BytesIO(response.content))
                        st.image(
                            image,
                            caption=img_data["caption"],
                            use_container_width=True
                        )
                    else:
                        # Load local image
                        image = Image.open(img_path)
                        st.image(
                            image,
                            caption=img_data["caption"],
                            use_container_width=True
                        )
                    
                    # Show tags
                    if img_data.get("tags"):
                        tags_str = " ".join([f"#{tag}" for tag in img_data["tags"]])
                        st.caption(f"🏷️ {tags_str}")
                        
                except:
                    # Silently skip any images that fail to load
                    continue

def stream_chat_response(query, language=None):
    """Stream chat response from API.
    
    Args:
        query (str): User query.
        language (str, optional): Language code.
        
    Yields:
        tuple: (text_chunk, images_data)
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={"query": query, "language": language},
            stream=True,
            timeout=30
        )
        
        if response.status_code == 200:
            images_data = None
            text_response = ""
            
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    # Check for image metadata
                    if "__IMAGES__" in chunk:
                        parts = chunk.split("__IMAGES__")
                        for i, part in enumerate(parts):
                            if i % 2 == 1:  # Odd indices contain JSON
                                try:
                                    img_info = json.loads(part)
                                    if img_info["type"] == "images":
                                        images_data = img_info["data"]
                                except json.JSONDecodeError:
                                    pass
                            elif part.strip():  # Even indices contain text
                                text_response += part
                                yield part, images_data
                    else:
                        text_response += chunk
                        yield chunk, images_data
        else:
            yield f"Error: {response.status_code}", None
            
    except Exception as e:
        yield f"Error: {str(e)}", None

def main():
    """Main Streamlit application."""
    st.title("🏖️🌴 NhaTrangGo 🤖")
    st.markdown("Your friendly guide to Nha Trang, Vietnam!")
    
    # Sidebar
    with st.sidebar:
        st.header("Sample Queries")
        st.markdown("""
        **Text Queries:**
        - "Tell me about Nha Trang beaches"
        - "Best restaurants in Nha Trang"
        - "Weather in Nha Trang today"
        
        **Image Queries:**
        - "Show me images of Nha Trang beaches"
        - "Pictures of sunset in Nha Trang"
        - "Photos of Po Nagar towers"
        """)
        
        if st.button("Clear Chat History"):
            try:
                requests.delete(f"{API_BASE_URL}/history")
                st.success("Chat history cleared!")
                st.rerun()
            except:
                st.error("Failed to clear history")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("images"):
                display_images(message["images"])

    # Chat input
    if prompt := st.chat_input("Ask me anything about Nha Trang..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            images_placeholder = st.empty()
            
            full_response = ""
            images_data = None
            
            # Stream response
            for chunk, images in stream_chat_response(prompt):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
                
                if images and not images_data:
                    images_data = images
                    with images_placeholder.container():
                        display_images(images_data)
            
            message_placeholder.markdown(full_response)
            
            # Add assistant message to history
            assistant_message = {
                "role": "assistant", 
                "content": full_response
            }
            if images_data:
                assistant_message["images"] = images_data
                
            st.session_state.messages.append(assistant_message)

if __name__ == "__main__":
    main()