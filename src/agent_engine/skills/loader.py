"""Skills file loader and integration."""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from agent_engine.config import get_settings
from agent_engine.tools.mcp_protocol import MCPParameterSchema, MCPToolSchema
from agent_engine.tools.registry import get_tool_registry


@dataclass
class Skill:
    """Represents a loaded skill."""

    name: str
    description: str
    instructions: str
    parameters: list[dict[str, Any]] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    version: str = "1.0.0"
    file_path: str = ""

    def to_tool_schema(self) -> MCPToolSchema:
        """Convert skill to MCP tool schema."""
        # Build parameter schema from skill parameters
        properties = {}
        required = []

        for param in self.parameters:
            param_name = param.get("name", "")
            if not param_name:
                continue

            properties[param_name] = {
                "type": param.get("type", "string"),
                "description": param.get("description", ""),
            }

            if param.get("required", False):
                required.append(param_name)

        return MCPToolSchema(
            name=self.name,
            description=self.description,
            parameters=MCPParameterSchema(
                type="object",
                properties=properties,
                required=required,
            ),
            tags=self.tags + ["skill"],
            version=self.version,
        )

    def get_prompt_context(self) -> str:
        """Get skill content formatted for prompt injection."""
        parts = [
            f"## Skill: {self.name}",
            f"Description: {self.description}",
            "",
            "### Instructions:",
            self.instructions,
        ]

        if self.examples:
            parts.append("")
            parts.append("### Examples:")
            for i, example in enumerate(self.examples, 1):
                parts.append(f"{i}. {example}")

        return "\n".join(parts)


class SkillLoader:
    """Loader for skill files (Markdown/YAML format)."""

    SKILL_FILE_PATTERNS = ["*.md", "*.yaml", "*.yml", "SKILL.md"]

    def __init__(self, skills_dir: str | None = None):
        """Initialize skill loader.

        Args:
            skills_dir: Directory containing skill files.
        """
        settings = get_settings()
        self.skills_dir = Path(skills_dir or settings.skills_dir)
        self._skills: dict[str, Skill] = {}

    def load_all(self) -> dict[str, Skill]:
        """Load all skills from the skills directory.

        Returns:
            Dictionary mapping skill names to Skill objects.
        """
        if not self.skills_dir.exists():
            return {}

        self._skills.clear()

        # Find all skill files
        for pattern in self.SKILL_FILE_PATTERNS:
            for file_path in self.skills_dir.rglob(pattern):
                try:
                    skill = self._load_file(file_path)
                    if skill:
                        self._skills[skill.name] = skill
                except Exception as e:
                    print(f"Error loading skill from {file_path}: {e}")

        return self._skills

    def load_file(self, file_path: str | Path) -> Skill | None:
        """Load a single skill file.

        Args:
            file_path: Path to the skill file.

        Returns:
            Loaded Skill or None if failed.
        """
        return self._load_file(Path(file_path))

    def _load_file(self, file_path: Path) -> Skill | None:
        """Load a skill from a file."""
        if not file_path.exists():
            return None

        content = file_path.read_text(encoding="utf-8")

        if file_path.suffix in (".yaml", ".yml"):
            return self._parse_yaml_skill(content, str(file_path))
        else:
            return self._parse_markdown_skill(content, str(file_path))

    def _parse_markdown_skill(self, content: str, file_path: str) -> Skill | None:
        """Parse a markdown skill file."""
        # Extract YAML frontmatter if present
        frontmatter = {}
        body = content

        frontmatter_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
        if frontmatter_match:
            try:
                frontmatter = yaml.safe_load(frontmatter_match.group(1))
                body = content[frontmatter_match.end():]
            except yaml.YAMLError:
                pass

        # Extract skill name from first header or frontmatter
        name = frontmatter.get("name", "")
        if not name:
            header_match = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
            if header_match:
                name = header_match.group(1).strip()
            else:
                # Use filename
                name = Path(file_path).stem

        # Clean name for use as tool name
        name = re.sub(r"[^a-zA-Z0-9_]", "_", name.lower())

        # Extract description
        description = frontmatter.get("description", "")
        if not description:
            # Use first paragraph after header
            desc_match = re.search(r"^#.*?\n\n(.+?)(?:\n\n|$)", body, re.DOTALL)
            if desc_match:
                description = desc_match.group(1).strip()

        # Extract examples section
        examples = frontmatter.get("examples", [])
        if not examples:
            examples_match = re.search(
                r"##\s*Examples?\s*\n(.*?)(?=\n##|\Z)", body, re.DOTALL | re.IGNORECASE
            )
            if examples_match:
                # Parse bullet points
                for line in examples_match.group(1).split("\n"):
                    line = line.strip()
                    if line.startswith(("-", "*", "•")):
                        examples.append(line[1:].strip())

        return Skill(
            name=name,
            description=description or f"Skill from {Path(file_path).name}",
            instructions=body,
            parameters=frontmatter.get("parameters", []),
            examples=examples,
            tags=frontmatter.get("tags", []),
            version=frontmatter.get("version", "1.0.0"),
            file_path=file_path,
        )

    def _parse_yaml_skill(self, content: str, file_path: str) -> Skill | None:
        """Parse a YAML skill file."""
        try:
            data = yaml.safe_load(content)
        except yaml.YAMLError:
            return None

        if not isinstance(data, dict):
            return None

        name = data.get("name", Path(file_path).stem)
        name = re.sub(r"[^a-zA-Z0-9_]", "_", name.lower())

        return Skill(
            name=name,
            description=data.get("description", f"Skill from {Path(file_path).name}"),
            instructions=data.get("instructions", ""),
            parameters=data.get("parameters", []),
            examples=data.get("examples", []),
            tags=data.get("tags", []),
            version=data.get("version", "1.0.0"),
            file_path=file_path,
        )

    def get_skill(self, name: str) -> Skill | None:
        """Get a skill by name.

        Args:
            name: The skill name.

        Returns:
            Skill or None if not found.
        """
        return self._skills.get(name)

    def list_skills(self, tags: list[str] | None = None) -> list[Skill]:
        """List loaded skills.

        Args:
            tags: Optional filter by tags.

        Returns:
            List of skills.
        """
        skills = list(self._skills.values())

        if tags:
            skills = [s for s in skills if any(t in s.tags for t in tags)]

        return skills

    def register_as_tools(self) -> int:
        """Register all loaded skills as tools.

        Returns:
            Number of skills registered.
        """
        registry = get_tool_registry()
        count = 0

        for skill in self._skills.values():
            schema = skill.to_tool_schema()

            # Create implementation that returns skill instructions
            async def skill_impl(
                skill_instructions: str = skill.instructions,
                **kwargs,
            ) -> dict:
                return {
                    "instructions": skill_instructions,
                    "parameters": kwargs,
                    "message": "Apply the skill instructions with the provided parameters.",
                }

            registry.register_tool(schema, skill_impl)
            count += 1

        return count

    def get_context_for_request(
        self,
        request: str,
        max_skills: int = 3,
    ) -> str:
        """Get relevant skill context for a request.

        Args:
            request: The user's request.
            max_skills: Maximum skills to include.

        Returns:
            Formatted context string.
        """
        # Simple keyword matching for now
        # In production, use semantic search
        request_lower = request.lower()
        scored_skills = []

        for skill in self._skills.values():
            score = 0
            # Check name match
            if skill.name in request_lower:
                score += 10

            # Check description match
            desc_lower = skill.description.lower()
            for word in request_lower.split():
                if len(word) > 3 and word in desc_lower:
                    score += 2

            # Check tag match
            for tag in skill.tags:
                if tag.lower() in request_lower:
                    score += 5

            if score > 0:
                scored_skills.append((score, skill))

        # Sort by score and take top N
        scored_skills.sort(key=lambda x: x[0], reverse=True)
        top_skills = [s for _, s in scored_skills[:max_skills]]

        if not top_skills:
            return ""

        context_parts = ["# Available Skills\n"]
        for skill in top_skills:
            context_parts.append(skill.get_prompt_context())
            context_parts.append("")

        return "\n".join(context_parts)


# Global loader instance
_loader: SkillLoader | None = None


def get_skill_loader() -> SkillLoader:
    """Get the global skill loader instance."""
    global _loader
    if _loader is None:
        _loader = SkillLoader()
    return _loader


def load_skills(skills_dir: str | None = None) -> dict[str, Skill]:
    """Load skills from a directory.

    Args:
        skills_dir: Directory containing skill files.

    Returns:
        Dictionary of loaded skills.
    """
    loader = get_skill_loader()
    if skills_dir:
        loader.skills_dir = Path(skills_dir)
    return loader.load_all()
