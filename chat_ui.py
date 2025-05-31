
import streamlit as st
import requests
import re

# API base URL (adjust if your FastAPI server is running on a different host/port)
API_BASE_URL = "http://localhost:8000"

def _fix_streamlit_space(text: str) -> str:
    """Fix silly streamlit issue where a newline needs 2 spaces before it.

    See https://github.com/streamlit/streamlit/issues/868
    """
    def _replacement(match: re.Match):
        # Check if the match is preceded by a space
        if match.group(0).startswith(" "):
            # If preceded by one space, add one more space
            return " \n"
        else:
            # If not preceded by any space, add two spaces
            return "  \n"
    return re.sub(r"( ?)\n", _replacement, text)

def main():
    """Main function to set up the Streamlit app with navigation.

    Displays a sidebar for navigation and loads the chat UI.
    """
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Chat"])
    if page == "Chat":
        display_chat()

def parse_history(history_str: str) -> list:
    """
    Parse the chat history string into a list of messages.
    Assumes format: "Human: query\nAssistant: response\n"
    """
    messages = []
    for line in history_str.strip().split("\n"):
        if line.startswith("Human: "):
            messages.append({"role": "user", "content": line[7:]})
        elif line.startswith("Assistant: "):
            messages.append({"role": "assistant", "content": line[11:]})
    return messages

def stream_response(query: str):
    """
    Send the chatbot's response request to the /chat endpoint and yield the full response.
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={"query": query},
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            yield data.get("response", "")
        else:
            yield f"Error: API returned status code {response.status_code}"
    except requests.exceptions.RequestException as e:
        yield f"Error connecting to API: {e}"

def display_chat():
    """Display the chat interface with history, input, and responses."""
    st.title("Nha Trang Chatbot 🤖🏖️🌴")

    # Use session_state to keep local chat history
    if "local_history" not in st.session_state:
        # On first load, fetch from backend
        try:
            response = requests.get(f"{API_BASE_URL}/history")
            if response.status_code == 200:
                history = response.json()
                history_str = history.get("chat_history", "")
                st.session_state["local_history"] = parse_history(history_str)
            else:
                st.session_state["local_history"] = []
        except requests.exceptions.RequestException:
            st.session_state["local_history"] = []

    # Display all messages in local history
    for msg in st.session_state["local_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=False)

    # Chat input for new queries
    if prompt := st.chat_input("What is your question?"):
        # Display user's message and append to local history
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state["local_history"].append({"role": "user", "content": prompt})

        # Show "working..." while waiting for the bot's response
        with st.chat_message("assistant"):
            working_placeholder = st.empty()
            working_placeholder.markdown("*working...*", unsafe_allow_html=True)
            full_response = next(stream_response(prompt))
            working_placeholder.markdown(_fix_streamlit_space(full_response), unsafe_allow_html=False)
        st.session_state["local_history"].append({"role": "assistant", "content": full_response})

    # Button to clear chat history
    if st.button("Clear History"):
        try:
            response = requests.delete(f"{API_BASE_URL}/history")
            if response.status_code == 200:
                st.session_state["local_history"] = []
                st.success("History cleared successfully.")
            else:
                st.error(f"Failed to clear history: Status code {response.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to API: {e}")

if __name__ == "__main__":
    main()