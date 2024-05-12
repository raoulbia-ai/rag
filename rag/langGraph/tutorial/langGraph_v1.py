"""https://medium.com/@cplog/introduction-to-langgraph-a-beginners-guide-14f9be027141"""

from langgraph.graph import StateGraph, END
from typing import Dict, TypedDict, Optional
from langchain_core.runnables.graph import CurveStyle, NodeColors, MermaidDrawMethod
# from IPython.display import display, HTML, Image

def classify(question=None):
    return "classified as X"

def classify_input_node(state):
    question = state.get('question', '').strip()
    classification = classify(question)  # Assume a function that classifies the input
    return {"classification": classification}

def handle_greeting_node(state):
    return {"response": "Hello! How can I help you today?"}

def handle_search_node(state):
    question = state.get('question', '').strip()
    search_result = f"Search result for '{question}'"
    return {"response": search_result}

def decide_next_node(state):
    if state.get('classification') == "greeting":
        return "handle_greeting" 
    else:
         return "handle_search"


class GraphState(TypedDict):
    question: Optional[str] = None
    classification: Optional[str] = None
    response: Optional[str] = None


workflow = StateGraph(GraphState)
workflow.add_node("classify_input", classify_input_node)
workflow.add_node("handle_greeting", handle_greeting_node)
workflow.add_node("handle_search", handle_search_node)  
workflow.add_conditional_edges(
    "classify_input",
    decide_next_node,
    {
        "handle_greeting": "handle_greeting",
        "handle_search": "handle_search"
    }
)
workflow.set_entry_point("classify_input")
workflow.add_edge('handle_greeting', END)
workflow.add_edge('handle_search', END)
 
G = workflow.compile()
inputs = {"question": "Hello, how are you?"}
result = G.invoke(inputs)
print(result)

with open('output.png', 'wb') as f:
    f.write(G.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API))