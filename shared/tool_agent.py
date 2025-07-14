import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Dict, List


@dataclass
class ToolResult:
    """Result from a tool execution"""
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time: Optional[float] = None


class CodeRetrievalTool:
    """Tool for retrieving code content by file path"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.name = "get_code_by_filepath"
        self.description = "Retrieve current code content from a specific file path"

    def execute(self, file_path: str) -> ToolResult:
        """Execute the code retrieval tool"""
        start_time = datetime.now()

        try:
            # Normalize path - handle both absolute and relative paths
            if not os.path.isabs(file_path):
                full_path = self.project_root / file_path.lstrip('/')
            else:
                full_path = Path(file_path)

            # Security check - ensure path is within project root
            try:
                full_path.resolve().relative_to(self.project_root.resolve())
            except ValueError:
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    result=None,
                    error=f"Path {file_path} is outside project root for security"
                )

            # Check if file exists
            if not full_path.exists():
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    result=None,
                    error=f"File not found: {file_path}"
                )

            # Read file content
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Get file metadata
            stat = full_path.stat()

            result = {
                "file_path": str(full_path),
                "relative_path": str(full_path.relative_to(self.project_root)),
                "content": content,
                "size_bytes": stat.st_size,
                "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "file_extension": full_path.suffix
            }

            execution_time = (datetime.now() - start_time).total_seconds()

            return ToolResult(
                tool_name=self.name,
                success=True,
                result=result,
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return ToolResult(
                tool_name=self.name,
                success=False,
                result=None,
                error=str(e),
                execution_time=execution_time
            )

    def get_tool_schema(self) -> Dict[str, Any]:
        """Get the tool schema for LLM function calling"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the code file to retrieve (can be relative or absolute)"
                    }
                },
                "required": ["file_path"]
            }
        }


class ToolAgent:
    """Agent that manages and executes tools"""

    def __init__(self, project_root: str):
        self.project_root = project_root
        self.tools = {}
        self._register_tools()

    def _register_tools(self):
        """Register available tools"""
        # Code retrieval tool
        code_tool = CodeRetrievalTool(self.project_root)
        self.tools[code_tool.name] = code_tool

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools for LLM"""
        return [tool.get_tool_schema() for tool in self.tools.values()]

    def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a specific tool with given parameters"""
        if tool_name not in self.tools:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=f"Unknown tool: {tool_name}"
            )

        return self.tools[tool_name].execute(**kwargs)