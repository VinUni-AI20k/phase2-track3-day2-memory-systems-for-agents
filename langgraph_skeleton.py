"""
Skeleton LangGraph implementation for Python 3.14 compatibility.

Mirrors the real LangGraph API surface used in this project:
  StateGraph, CompiledGraph, END sentinel.

Real LangGraph (langgraph>=0.1) is not yet available for Python 3.14,
so this thin skeleton keeps the agent code identical to what would run
on a supported runtime.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type


END = "__end__"


class StateGraph:
    """Skeleton StateGraph that mimics langgraph.graph.StateGraph."""

    def __init__(self, state_schema: Type) -> None:
        self.state_schema = state_schema
        self._nodes: Dict[str, Callable] = {}
        self._edges: Dict[str, str] = {}
        self._conditional_edges: Dict[str, tuple] = {}
        self._entry_point: Optional[str] = None

    def add_node(self, name: str, func: Callable) -> None:
        self._nodes[name] = func

    def add_edge(self, from_node: str, to_node: str) -> None:
        self._edges[from_node] = to_node

    def add_conditional_edges(
        self,
        from_node: str,
        condition: Callable,
        mapping: Dict[str, str],
    ) -> None:
        self._conditional_edges[from_node] = (condition, mapping)

    def set_entry_point(self, node_name: str) -> None:
        self._entry_point = node_name

    def compile(self) -> "CompiledGraph":
        if self._entry_point is None:
            raise ValueError("Entry point not set. Call set_entry_point() first.")
        return CompiledGraph(self)


class CompiledGraph:
    """Executable graph produced by StateGraph.compile()."""

    def __init__(self, graph: StateGraph) -> None:
        self._graph = graph

    def invoke(self, state: Dict[str, Any], config: Optional[Dict] = None) -> Dict[str, Any]:
        current = self._graph._entry_point
        max_steps = 100
        step = 0

        while current != END and step < max_steps:
            step += 1

            if current not in self._graph._nodes:
                raise KeyError(f"Node '{current}' not found in graph.")

            state = self._graph._nodes[current](state)

            if current in self._graph._conditional_edges:
                condition_fn, mapping = self._graph._conditional_edges[current]
                key = condition_fn(state)
                current = mapping.get(key, END)
            elif current in self._graph._edges:
                current = self._graph._edges[current]
            else:
                break

        return state
