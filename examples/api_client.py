"""Example API client for the Agent Engine.

This shows how to interact with the Agent Engine via HTTP API.
"""

import asyncio
import httpx


BASE_URL = "http://localhost:8000"


async def check_health():
    """Check API health."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/api/v1/health")
        print("Health Check:")
        print(response.json())
        return response.status_code == 200


async def list_tools():
    """List available tools."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/api/v1/tools")
        data = response.json()
        print(f"\nAvailable Tools ({data['total']}):")
        for tool in data["tools"]:
            print(f"  - {tool['name']}: {tool['description']}")


async def create_task(request: str):
    """Create and run a task."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        print(f"\nCreating task: {request}")

        response = await client.post(
            f"{BASE_URL}/api/v1/tasks",
            json={
                "request": request,
                "max_tokens": 50000,
                "max_steps": 20,
            },
        )

        if response.status_code == 201:
            data = response.json()
            print(f"Task ID: {data['id']}")
            print(f"Status: {data['status']}")
            print(f"Tokens used: {data['metrics']['total_tokens']}")
            print(f"Steps: {data['metrics']['step_count']}")

            if data.get("subtasks"):
                print("\nSubtasks:")
                for st in data["subtasks"]:
                    print(f"  - [{st.get('status', 'unknown')}] {st.get('description', 'N/A')}")

            return data["id"]
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None


async def get_task_status(task_id: str):
    """Get task status."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/api/v1/tasks/{task_id}/status")

        if response.status_code == 200:
            data = response.json()
            print(f"\nTask {task_id}:")
            print(f"  Status: {data['status']}")
            print(f"  Steps: {data['step_count']}")
            print(f"  Tokens: {data['total_tokens']}")
            if data.get("error"):
                print(f"  Error: {data['error']}")
        else:
            print(f"Task not found: {task_id}")


async def call_tool_directly(tool_name: str, arguments: dict):
    """Call a tool directly."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/api/v1/tools/call",
            json={
                "tool_name": tool_name,
                "arguments": arguments,
            },
        )

        data = response.json()
        print(f"\nTool Call: {tool_name}")
        print(f"  Status: {data['status']}")
        print(f"  Result: {data.get('result')}")
        print(f"  Time: {data['execution_time_ms']:.2f}ms")

        if data.get("error"):
            print(f"  Error: {data['error']}")


async def store_memory(content: str, content_type: str = "text"):
    """Store content in long-term memory."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/api/v1/memory",
            json={
                "content": content,
                "content_type": content_type,
                "importance": 0.7,
            },
        )

        if response.status_code == 201:
            data = response.json()
            print(f"\nMemory stored with ID: {data['id']}")
        else:
            print(f"Failed to store memory: {response.text}")


async def search_memory(query: str):
    """Search long-term memory."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/api/v1/memory/search",
            json={
                "query": query,
                "limit": 5,
                "min_similarity": 0.3,
            },
        )

        data = response.json()
        print(f"\nMemory search: '{query}'")
        print(f"Found {data['total']} results:")

        for result in data["results"]:
            print(f"  - [{result['content_type']}] (sim: {result['similarity']:.3f})")
            print(f"    {result['content'][:100]}...")


async def main():
    """Main function demonstrating API usage."""
    print("🔌 Agent Engine API Client Example")
    print("=" * 50)

    # Check if server is running
    try:
        if not await check_health():
            print("\n❌ Server not healthy")
            return
    except httpx.ConnectError:
        print(f"\n❌ Cannot connect to server at {BASE_URL}")
        print("   Start the server with: poetry run agent-engine")
        return

    print("\n✅ Server is running!")

    # List tools
    await list_tools()

    # Call a tool directly
    await call_tool_directly(
        "python_eval",
        {"expression": "sum(range(1, 101))"},
    )

    # Create a task
    task_id = await create_task("What is 2 + 2?")

    if task_id:
        # Check status
        await get_task_status(task_id)

    print("\n✅ API client example completed!")


if __name__ == "__main__":
    asyncio.run(main())
