"""Tests for API routes."""

import pytest
from httpx import AsyncClient, ASGITransport

from agent_engine.main import app


@pytest.fixture
async def client():
    """Create test client."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as ac:
        yield ac


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_check(self, client):
        """Test health check returns OK."""
        response = await client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestToolEndpoints:
    """Tests for tool endpoints."""

    @pytest.mark.asyncio
    async def test_list_tools(self, client):
        """Test listing available tools."""
        response = await client.get("/api/v1/tools")

        assert response.status_code == 200
        data = response.json()
        assert "tools" in data
        assert "total" in data
        assert isinstance(data["tools"], list)

    @pytest.mark.asyncio
    async def test_register_tool(self, client):
        """Test registering a new tool."""
        tool_data = {
            "name": "test_api_tool",
            "description": "A test tool from API",
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {"type": "string"}
                },
                "required": ["input"]
            },
            "tags": ["test", "api"]
        }

        response = await client.post("/api/v1/tools", json=tool_data)

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "test_api_tool"
        assert data["description"] == "A test tool from API"

    @pytest.mark.asyncio
    async def test_register_duplicate_tool(self, client):
        """Test registering duplicate tool fails."""
        tool_data = {
            "name": "duplicate_tool",
            "description": "First registration",
        }

        # First registration
        response = await client.post("/api/v1/tools", json=tool_data)
        assert response.status_code == 201

        # Duplicate registration
        response = await client.post("/api/v1/tools", json=tool_data)
        assert response.status_code == 400


class TestRootEndpoint:
    """Tests for root endpoint."""

    @pytest.mark.asyncio
    async def test_root(self, client):
        """Test root endpoint."""
        response = await client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Agent Engine"
        assert "version" in data
        assert "docs" in data
