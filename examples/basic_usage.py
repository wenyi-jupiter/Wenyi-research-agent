"""Basic usage example for the Agent Engine.

This example demonstrates how to use the agent engine programmatically.
"""

import asyncio
import os
import sys

# Add src to path for direct execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_engine.agents import AgentOrchestrator, create_initial_state
from agent_engine.tools import get_tool_registry, tool
from agent_engine.config import get_settings


# Register a custom tool
@tool(
    name="calculator",
    description="Perform basic arithmetic calculations",
    tags=["math", "utility"],
)
async def calculator(expression: str) -> dict:
    """Evaluate a mathematical expression.

    Args:
        expression: Math expression like "2 + 2" or "10 * 5"

    Returns:
        Dictionary with result or error
    """
    try:
        # Safe eval for basic math
        allowed_chars = set("0123456789+-*/().% ")
        if not all(c in allowed_chars for c in expression):
            return {"error": "Invalid characters in expression"}

        result = eval(expression)
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"expression": expression, "error": str(e)}


async def run_simple_task():
    """Run a simple task without database persistence."""
    print("=" * 60)
    print("Running Simple Task")
    print("=" * 60)

    # Create orchestrator without checkpointing
    orchestrator = AgentOrchestrator(checkpointer=None)

    # Run a task
    user_request = "Calculate 15 * 7 + 23 and tell me the result"

    print(f"\nUser Request: {user_request}\n")

    try:
        final_state = await orchestrator.run(
            user_request=user_request,
            task_id="example_task_001",
        )

        print("\n--- Task Completed ---")
        print(f"Status: {final_state.get('status')}")
        print(f"Steps: {final_state.get('metrics', {}).get('step_count', 0)}")
        print(f"Tokens: {final_state.get('metrics', {}).get('total_tokens', 0)}")

        print("\n--- Subtasks ---")
        for subtask in final_state.get("subtasks", []):
            print(f"  - [{subtask.get('status')}] {subtask.get('description')}")
            if subtask.get("result"):
                print(f"    Result: {subtask.get('result')}")

    except Exception as e:
        print(f"Error: {e}")


async def run_streaming_task():
    """Run a task with streaming output."""
    print("\n" + "=" * 60)
    print("Running Streaming Task")
    print("=" * 60)

    orchestrator = AgentOrchestrator(checkpointer=None)

    user_request = "List the first 5 prime numbers"

    print(f"\nUser Request: {user_request}\n")
    print("Streaming execution:")

    try:
        async for event in orchestrator.stream(
            user_request=user_request,
            task_id="example_streaming_001",
        ):
            for node_name, state in event.items():
                status = state.get("status", "")
                step = state.get("metrics", {}).get("step_count", 0)
                print(f"  [{node_name}] Step {step} - Status: {status}")

    except Exception as e:
        print(f"Error: {e}")


async def demonstrate_tool_registry():
    """Demonstrate the tool registry."""
    print("\n" + "=" * 60)
    print("Tool Registry Demo")
    print("=" * 60)

    registry = get_tool_registry()

    # List all registered tools
    tools = registry.list_tools()
    print(f"\nRegistered Tools ({len(tools)}):")
    for t in tools:
        print(f"  - {t.name}: {t.description}")
        print(f"    Tags: {t.tags}")

    # Get tools by tag
    math_tools = registry.list_tools(tags=["math"])
    print(f"\nMath Tools: {[t.name for t in math_tools]}")


async def main():
    """Main entry point."""
    print("\n🚀 Agent Engine - Basic Usage Examples\n")

    # Check for API keys
    settings = get_settings()
    if not settings.minimax_api_key:
        print("⚠️  Warning: No Minimax API key configured.")
        print("   Set MINIMAX_API_KEY and MINIMAX_GROUP_ID in .env file")
        print("   Running tool registry demo only...\n")
        await demonstrate_tool_registry()
        return

    # Run examples
    await demonstrate_tool_registry()

    try:
        await run_simple_task()
    except Exception as e:
        print(f"Simple task failed: {e}")

    try:
        await run_streaming_task()
    except Exception as e:
        print(f"Streaming task failed: {e}")

    print("\n✅ Examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
