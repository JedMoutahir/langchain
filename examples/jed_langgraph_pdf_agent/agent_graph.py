import argparse, os, json
from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

# --- Tool: Retriever --------------------------------------------------------

class RetrieverTool:
    def __init__(self, index_dir: str, k: int = 4):
        self.db = FAISS.load_local(index_dir, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
        self.k = k
    def __call__(self, query: str) -> List[Dict[str, Any]]:
        docs: List[Document] = self.db.similarity_search(query, k=self.k)
        return [{"content": d.page_content, "metadata": d.metadata} for d in docs]

# --- Graph State ------------------------------------------------------------

class GraphState(BaseModel):
    question: str
    context: List[Dict[str, Any]] = Field(default_factory=list)
    answer: str = ""

# --- Nodes ------------------------------------------------------------------

def route_node(state: GraphState) -> str:
    # simple router: if we have no context yet, retrieve; else answer
    return "retrieve" if not state.context else "answer"

def retrieve_node(state: GraphState, tools) -> GraphState:
    ctx = tools["retriever"](state.question)
    return GraphState(question=state.question, context=ctx, answer=state.answer)

def answer_node(state: GraphState, llm) -> GraphState:
    system = "You are a precise assistant. Use the provided context to answer the user's question. Cite sources with (p#:filename) when possible."
    context_text = "\n\n".join(
        f"- {c['content']}\n  META: {c.get('metadata',{})}" for c in state.context
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("user", "Question: {q}\n\nContext:\n{ctx}\n\nAnswer concisely and cite."),
    ]).format(q=state.question, ctx=context_text)

    resp = llm.invoke(prompt)
    return GraphState(question=state.question, context=state.context, answer=resp.content)

# --- Build Graph ------------------------------------------------------------

def build_graph(index_dir: str):
    tools = {"retriever": RetrieverTool(index_dir=index_dir, k=4)}
    llm = ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
        temperature=0.2,
    )

    g = StateGraph(GraphState)
    g.add_node("retrieve", lambda s: retrieve_node(s, tools))
    g.add_node("answer", lambda s: answer_node(s, llm))
    g.set_entry_point("router")
    g.add_node("router", route_node)
    g.add_conditional_edges("router", lambda s: route_node(s), {"retrieve":"retrieve","answer":"answer"})
    g.add_edge("retrieve", "answer")
    g.add_edge("answer", END)
    return g.compile()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--question", required=True)
    args = ap.parse_args()

    graph = build_graph(args.index_dir)
    out = graph.invoke(GraphState(question=args.question))
    print(out.answer)

if __name__ == "__main__":
    main()
