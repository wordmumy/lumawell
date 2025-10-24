# main.py
import os
from dotenv import load_dotenv

# 先加载 .env
load_dotenv(override=True)

from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import HumanMessage
from graph.nodes import (
    AgentState, router_node, rag_node, tool_node,
    fitness_agent_node, nutrition_agent_node, environment_agent_node,
    mind_agent_node, medical_agent_node, safety_node,
    greet_node,    
    idle_node,      
)

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    import sqlite3
    # 直接传入 sqlite3 连接，避免 from_conn_string 返回 context manager 的坑
    conn = sqlite3.connect("healthbot.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
except Exception as e:
    from langgraph.checkpoint.memory import MemorySaver
    print(f"[checkpoint] fallback to MemorySaver: {type(e).__name__}: {e}")
    checkpointer = MemorySaver()

def build_graph():
    g = StateGraph(AgentState)

    # 节点注册
    g.add_node("router", router_node)
    g.add_node("tools", tool_node)             # bmi/tdee/skin
    g.add_node("fitness", fitness_agent_node)  
    g.add_node("nutrition", nutrition_agent_node)
    g.add_node("environment", environment_agent_node)
    g.add_node("mind", mind_agent_node)
    g.add_node("medical", medical_agent_node)
    g.add_node("rag", rag_node)
    g.add_node("safety", safety_node)

    g.add_node("greet", greet_node)  
    g.add_node("idle", idle_node)     

    g.add_edge(START, "router")

    def route_decider(state: AgentState):
        r = state.get("route", "rag")
        if r in ("bmi","tdee","skin"):
            return "tools"
        if r == "fitness": return "fitness"
        if r == "nutrition": return "nutrition"
        if r == "environment": return "environment"
        if r == "mind": return "mind"
        if r == "medical": return "medical"
        if r == "safety": return "safety"
        if r == "greet": return "greet"  
        if r == "idle":  return "idle"   
        return "rag"

    g.add_conditional_edges("router", route_decider, {
        "tools": "tools",
        "fitness": "fitness",
        "nutrition": "nutrition",
        "environment": "environment",
        "mind": "mind",
        "medical": "medical",
        "rag": "rag",
        "safety": "safety",
        "greet": "greet",   
        "idle": "idle",    
    })

    # 工具/Agent 直接到 END
    g.add_edge("tools", END)
    g.add_edge("fitness", END)
    g.add_edge("nutrition", END)
    g.add_edge("environment", END)
    g.add_edge("mind", END)
    g.add_edge("medical", END)
    g.add_edge("rag", END)
    g.add_edge("safety", END)
    g.add_edge("greet", END) 
    g.add_edge("idle", END)  
    return g.compile(checkpointer=checkpointer)

def main():
    app = build_graph()
    print("=== LangGraph Health & Wellness Advisor ===")
    print("🚀 【Start】:/reset 重置；/exit 退出；/thread 查看线程ID\n")

    thread_id = "default"
    while True:
        q = input("🧐 【User】 :\n ")
        if q.strip() == "/exit": break
        if q.strip() == "/reset":
            import secrets
            thread_id = secrets.token_hex(4)
            print(f"[*] 会话已重置, ID={thread_id}")
            continue

        inputs = MessagesState(messages=[HumanMessage(content=q)])
        result = app.invoke(inputs, config={"configurable": {"thread_id": thread_id}})
        last = result["messages"][-1]
        content = getattr(last, "content", last.get("content") if isinstance(last, dict) else str(last))
        print("\n🤖 【LumaWell】 :\n", content, "\n")
if __name__ == "__main__":
    main()