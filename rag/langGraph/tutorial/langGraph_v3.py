from typing import Dict, TypedDict, Optional
from langgraph.graph import StateGraph, END
import random  # Added for generating random greetings

class GraphState(TypedDict):
    question: Optional[str] = None
    classification: Optional[str] = None
    response: Optional[str] = None

# Improved classify function to handle more types of questions dynamically
def classify(question: str) -> str:
    if "hello" in question.lower() or "hi" in question.lower():
        return "greeting"
    elif "how" in question.lower() or "what" in question.lower():
        return "info"
    else:
        return "unknown"

def classify_input_node(state: GraphState) -> Dict:
    question = state.get('question', '').strip()
    classification = classify(question)
    return {"classification": classification}

# Added new node for handling unknown queries
def handle_unknown_node(state: GraphState) -> Dict:
    return {"response": "Sorry, I didn't understand that. Can you please rephrase?"}

# Updated greeting handler to provide random greetings
def handle_greeting_node(state: GraphState) -> Dict:
    greetings = ["Hello! How can I help you today?", "Hi there! What can I do for you?", "Greetings! How may I assist?"]
    return {"response": random.choice(greetings)}

# Existing function for handling search, simplified example assumes all non-greeting questions are search
def handle_search_node(state: GraphState) -> Dict:
    question = state.get('question', '').strip()
    search_result = f"I found information related to '{question}'. Would you like more details?"
    return {"response": search_result}

def decide_next_node(state: GraphState) -> str:
    classification = state.get('classification')
    if classification == "greeting":
        return "handle_greeting"
    elif classification == "info":
        return "handle_search"
    else:
        return "handle_unknown"

workflow = StateGraph(GraphState)
workflow.add_node("classify_input", classify_input_node)
workflow.add_node("handle_greeting", handle_greeting_node)
workflow.add_node("handle_search", handle_search_node)
workflow.add_node("handle_unknown", handle_unknown_node)  # New node added for unknown inputs
workflow.add_conditional_edges(
    "classify_input",
    decide_next_node,
    {
        "handle_greeting": "handle_greeting",
        "handle_search": "handle_search",
        "handle_unknown": "handle_unknown"  # New edge for unknown path
    }
)
workflow.set_entry_point("classify_input")
workflow.add_edge('handle_greeting', END)
workflow.add_edge('handle_search', END)
workflow.add_edge('handle_unknown', END)  # Ensures the graph ends after handling unknown

# Compile the graph
G = workflow.compile()
# Test the graph with a sample input
query = "Hello, how are you?"
query = "blablabla"
inputs = {"question": query}
result = G.invoke(inputs)
print(result)


# from langchain_core.runnables.graph import CurveStyle, NodeColors, MermaidDrawMethod
# with open('output_v3.png', 'wb') as f:
#     f.write(G.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API))