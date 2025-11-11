import json
import shutil
import subprocess
from pathlib import Path

import litellm
from docker_utils import run_command_in_container
from litellm import completion

# Drop unsupported params for model compatibility
litellm.drop_params = True

# Estimate token count for context management
def estimate_tokens(text: str) -> int:
    """Rough estimate of token count (conservative estimate: 4 chars per token)"""
    return len(text) // 4

def estimate_message_tokens(messages: list) -> int:
    """Estimate total tokens in message list"""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            # Handle tool results that might be lists
            total += estimate_tokens(str(content))
    return total


class EnhancedTools:
    """Enhanced tools for advanced agent operations"""

    tools_info = [
        {
            "type": "function",
            "function": {
                "name": "codebase_search",
                "description": "Find snippets of code from the codebase most relevant to the search query. This is a semantic search tool, so the query should ask for something semantically matching what is needed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "explanation": {
                            "type": "string",
                            "description": "One sentence explanation as to why this tool is being used, and how it contributes to the goal.",
                        },
                        "query": {
                            "type": "string",
                            "description": "The search query to find relevant code. You should reuse the user's exact query/most recent message with their wording unless there is a clear reason not to.",
                        },
                        "target_directories": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Glob patterns for directories to search over",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the contents of a file. the output of this tool call will be the 1-indexed file contents from start_line_one_indexed to end_line_one_indexed_inclusive, together with a summary of the lines outside start_line_one_indexed and end_line_one_indexed_inclusive.Note: you are NOT allowed to read files in the /task directory or the ./run_tests.sh file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_file": {
                            "type": "string",
                            "description": "The path of the file to read. You can use either a relative path in the workspace or an absolute path.",
                        },
                        "explanation": {
                            "type": "string",
                            "description": "One sentence explanation as to why this tool is being used, and how it contributes to the goal.",
                        },
                        "should_read_entire_file": {
                            "type": "boolean",
                            "description": "Whether to read the entire file. Defaults to false.",
                        },
                        "start_line_one_indexed": {
                            "type": "integer",
                            "description": "The one-indexed line number to start reading from (inclusive).",
                        },
                        "end_line_one_indexed_inclusive": {
                            "type": "integer",
                            "description": "The one-indexed line number to end reading at (inclusive).",
                        },
                    },
                    "required": [
                        "target_file",
                        "should_read_entire_file",
                        "start_line_one_indexed",
                        "end_line_one_indexed_inclusive",
                    ],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_terminal_cmd",
                "description": "PROPOSE a command to run on behalf of the user. If you have this tool, note that you DO have the ability to run commands directly on the USER's system.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The terminal command to execute",
                        },
                        "explanation": {
                            "type": "string",
                            "description": "One sentence explanation as to why this command needs to be run and how it contributes to the goal.",
                        },
                        "is_background": {
                            "type": "boolean",
                            "description": "Whether the command should be run in the background",
                        },
                    },
                    "required": ["command", "is_background"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_dir",
                "description": "List the contents of a directory. The quick tool to use for discovery, before using more targeted tools like semantic search or file reading.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "relative_workspace_path": {
                            "type": "string",
                            "description": "Path to list contents of, relative to the workspace root.",
                        },
                        "explanation": {
                            "type": "string",
                            "description": "One sentence explanation as to why this tool is being used, and how it contributes to the goal.",
                        },
                    },
                    "required": ["relative_workspace_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "grep_search",
                "description": "This is best for finding exact text matches or regex patterns. Use this tool to run fast, exact regex searches over text files using the ripgrep engine.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The regex pattern to search for",
                        },
                        "explanation": {
                            "type": "string",
                            "description": "One sentence explanation as to why this tool is being used, and how it contributes to the goal.",
                        },
                        "case_sensitive": {
                            "type": "boolean",
                            "description": "Whether the search should be case sensitive",
                        },
                        "include_pattern": {
                            "type": "string",
                            "description": "Glob pattern for files to include (e.g. '*.ts' for TypeScript files)",
                        },
                        "exclude_pattern": {
                            "type": "string",
                            "description": "Glob pattern for files to exclude",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Edit a file using structured line-based edits. Supports both creating new files and modifying existing ones.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_file": {
                            "type": "string",
                            "description": "The target file to modify. You can use either a relative path in the workspace or an absolute path.",
                        },
                        "instructions": {
                            "type": "string",
                            "description": "A clear description of what changes are being made to the file.",
                        },
                        "edit_type": {
                            "type": "string",
                            "enum": ["line_edits"],
                            "description": "Type of edit to perform. Only 'line_edits' for structured line-based changes is supported.",
                        },
                        "line_edits": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {
                                        "type": "string",
                                        "enum": ["replace", "insert", "delete"],
                                        "description": "Type of line operation",
                                    },
                                    "line_number": {
                                        "type": "integer",
                                        "description": "1-indexed line number where the edit should be applied",
                                    },
                                    "content": {
                                        "type": "string",
                                        "description": "New content for replace/insert operations (not needed for delete)",
                                    },
                                },
                                "required": ["type", "line_number"],
                            },
                            "description": "Array of line-based edits to apply (required if edit_type is 'line_edits'). Edits are applied in reverse line order to avoid offset issues.",
                        },
                    },
                    "required": ["target_file", "instructions", "edit_type"],
                },
            },
        },
        # Intentionally omit search_replace from advertised tools to nudge structural edits via edit_file
        {
            "type": "function",
            "function": {
                "name": "file_search",
                "description": "Fast file search based on fuzzy matching against file path. Use if you know part of the file path but don't know where it's located exactly.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Fuzzy filename to search for",
                        },
                        "explanation": {
                            "type": "string",
                            "description": "One sentence explanation as to why this tool is being used, and how it contributes to the goal.",
                        },
                    },
                    "required": ["query", "explanation"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "delete_file",
                "description": "Deletes a file at the specified path.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_file": {
                            "type": "string",
                            "description": "The path of the file to delete, relative to the workspace root.",
                        },
                        "explanation": {
                            "type": "string",
                            "description": "One sentence explanation as to why this tool is being used, and how it contributes to the goal.",
                        },
                    },
                    "required": ["target_file"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for real-time information about any topic.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_term": {
                            "type": "string",
                            "description": "The search term to look up on the web. Be specific and include relevant keywords for better results.",
                        },
                        "explanation": {
                            "type": "string",
                            "description": "One sentence explanation as to why this tool is being used, and how it contributes to the goal.",
                        },
                    },
                    "required": ["search_term"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_diagram",
                "description": "Creates a Mermaid diagram that will be rendered in the chat UI.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Raw Mermaid diagram definition (e.g., 'graph TD; A-->B;').",
                        }
                    },
                    "required": ["content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "edit_notebook",
                "description": "Use this tool to edit a jupyter notebook cell.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_notebook": {
                            "type": "string",
                            "description": "The path to the notebook file you want to edit.",
                        },
                        "cell_idx": {
                            "type": "number",
                            "description": "The index of the cell to edit (0-based)",
                        },
                        "is_new_cell": {
                            "type": "boolean",
                            "description": "If true, a new cell will be created at the specified cell index. If false, the cell at the specified cell index will be edited.",
                        },
                        "cell_language": {
                            "type": "string",
                            "description": "The language of the cell to edit. Should be STRICTLY one of these: 'python', 'markdown', 'javascript', 'typescript', 'r', 'sql', 'shell', 'raw' or 'other'.",
                        },
                        "old_string": {
                            "type": "string",
                            "description": "The text to replace (must be unique within the cell, and must match the cell contents exactly, including all whitespace and indentation).",
                        },
                        "new_string": {
                            "type": "string",
                            "description": "The edited text to replace the old_string or the content for the new cell.",
                        },
                    },
                    "required": [
                        "target_notebook",
                        "cell_idx",
                        "is_new_cell",
                        "cell_language",
                        "old_string",
                        "new_string",
                    ],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "api_call",
                "description": "Make HTTP requests to test API endpoints. Useful for testing backend functionality and API integration.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "method": {
                            "type": "string",
                            "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"],
                            "description": "HTTP method to use"
                        },
                        "url": {
                            "type": "string",
                            "description": "The API endpoint URL (can be relative like '/api/users' or absolute)"
                        },
                        "headers": {
                            "type": "object",
                            "description": "HTTP headers to include in the request"
                        },
                        "body": {
                            "type": "object",
                            "description": "Request body for POST/PUT/PATCH requests"
                        },
                        "explanation": {
                            "type": "string",
                            "description": "One sentence explanation of why this API call is being made"
                        }
                    },
                    "required": ["method", "url", "explanation"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "database_query",
                "description": "Execute database queries to test data persistence and retrieval. Useful for verifying backend data operations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query_type": {
                            "type": "string",
                            "enum": ["find", "insert", "update", "delete", "aggregate"],
                            "description": "Type of database operation"
                        },
                        "collection": {
                            "type": "string",
                            "description": "MongoDB collection name"
                        },
                        "query": {
                            "type": "object",
                            "description": "Query object (MongoDB syntax)"
                        },
                        "data": {
                            "type": "object",
                            "description": "Data to insert/update (for insert/update operations)"
                        },
                        "explanation": {
                            "type": "string",
                            "description": "One sentence explanation of why this database operation is needed"
                        }
                    },
                    "required": ["query_type", "collection", "explanation"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "websocket_test",
                "description": "Test WebSocket/Socket.IO functionality for real-time features like chat, notifications, or live updates.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "event_name": {
                            "type": "string",
                            "description": "Socket.IO event name to test"
                        },
                        "event_data": {
                            "type": "object",
                            "description": "Data to send with the event"
                        },
                        "expected_response": {
                            "type": "string",
                            "description": "Expected response event or behavior"
                        },
                        "timeout": {
                            "type": "number",
                            "description": "Timeout in seconds (default: 10)"
                        },
                        "explanation": {
                            "type": "string",
                            "description": "One sentence explanation of what real-time functionality is being tested"
                        }
                    },
                    "required": ["event_name", "explanation"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "ui_test",
                "description": "Test frontend UI functionality by simulating user interactions and checking page state.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["screenshot", "click", "type", "navigate", "wait_for_element", "get_text"],
                            "description": "UI action to perform"
                        },
                        "selector": {
                            "type": "string",
                            "description": "CSS selector for the element (required for click, type, wait_for_element, get_text)"
                        },
                        "text": {
                            "type": "string",
                            "description": "Text to type (for type action)"
                        },
                        "url": {
                            "type": "string",
                            "description": "URL to navigate to (for navigate action)"
                        },
                        "timeout": {
                            "type": "number",
                            "description": "Timeout in seconds (default: 10)"
                        },
                        "explanation": {
                            "type": "string",
                            "description": "One sentence explanation of what UI functionality is being tested"
                        }
                    },
                    "required": ["action", "explanation"]
                }
            }
        },
    ]

    def __init__(self, container=None, base_path: str = "/", mern_config: dict = None):
        self.container = container
        self.base_path = Path(base_path)
        self.mern_config = mern_config or {
            "api_base_url": "http://localhost:5001",
            "frontend_url": "http://localhost:3000",
            "mongo_uri": "mongodb://localhost:27017/dev-arena",
            "websocket_url": "http://localhost:5001"
        }

    def codebase_search(
        self, query: str, explanation: str = "", target_directories: list = None
    ) -> dict:
        """Enhanced codebase search with semantic and text-based search"""
        try:
            results = []
            search_dirs = target_directories if target_directories else ["."]

            # Use ripgrep for better search if available, otherwise fallback to grep
            search_terms = query.lower().split()

            for search_dir in search_dirs:
                if self.container:
                    # First try ripgrep for content search
                    rg_cmd = ["rg", "-i", "--type", "py", "--type", "js", "--type", "ts",
                             "-n", "-C", "2", query, str(self.base_path / search_dir)]

                    result = run_command_in_container(self.container, rg_cmd)
                    if result["success"] and result["output"]:
                        # Parse ripgrep output
                        lines = result["output"].split("\n")
                        current_file = None
                        for line in lines:
                            if line and not line.startswith("--"):
                                if ":" in line:
                                    file_path, line_num, content = line.split(":", 2)
                                    if file_path != current_file:
                                        current_file = file_path
                                        results.append({
                                            "file": file_path,
                                            "relevance": 0.9,
                                            "snippet": content.strip(),
                                            "line_number": line_num
                                        })

                    # Fallback to find command for file discovery
                    if not results:
                        cmd = [
                            "find", str(self.base_path / search_dir), "-type", "f",
                            "(", "-name", "*.py", "-o", "-name", "*.js", "-o", "-name", "*.ts",
                            "-o", "-name", "*.jsx", "-o", "-name", "*.tsx", ")",
                            "-exec", "grep", "-l", "-i", query, "{}", ";"
                        ]
                        result = run_command_in_container(self.container, cmd)
                        if result["success"]:
                            files = [f.strip() for f in result["output"].split("\n") if f.strip()]
                            for file in files[:10]:
                                results.append({
                                    "file": file,
                                    "relevance": 0.7,
                                    "snippet": f"Contains '{query}'"
                                })
                else:
                    # Local search with better file content analysis
                    search_path = self.base_path / search_dir
                    for ext in ["*.py", "*.js", "*.ts", "*.jsx", "*.tsx", "*.java", "*.cpp", "*.c"]:
                        for file in search_path.glob(f"**/{ext}"):
                            if len(results) >= 10:
                                break
                            try:
                                with open(file, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                    if any(term in content.lower() for term in search_terms):
                                        # Find the best matching line
                                        lines = content.split('\n')
                                        best_line = ""
                                        best_score = 0
                                        for line in lines:
                                            score = sum(1 for term in search_terms if term in line.lower())
                                            if score > best_score:
                                                best_score = score
                                                best_line = line.strip()

                                        results.append({
                                            "file": str(file.relative_to(self.base_path)),
                                            "relevance": min(0.9, 0.6 + best_score * 0.1),
                                            "snippet": best_line or f"Found {best_score} matches in {file.name}"
                                        })
                            except (UnicodeDecodeError, PermissionError):
                                continue

            # Sort by relevance
            results.sort(key=lambda x: x["relevance"], reverse=True)
            return {"success": True, "results": results[:10], "explanation": explanation}
        except Exception as e:
            return {"success": False, "error": f"Codebase search failed: {str(e)}"}

    def read_file(
        self,
        target_file: str,
        explanation: str = "",
        should_read_entire_file: bool = True,
        start_line_one_indexed: int = 1,
        end_line_one_indexed_inclusive: int = -1,
    ) -> dict:
        """Read contents of a file with line range support"""
        try:
            if self.container:
                result = run_command_in_container(
                    container=self.container,
                    command=["cat", str(self.base_path / target_file)],
                )
                if result["success"]:
                    content = result["output"]
                else:
                    return {
                        "success": False,
                        "error": f"Failed to read file: {result.get('error', 'Unknown error')}",
                    }
            else:
                # Local file system
                full_path = self.base_path / target_file
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()

            lines = content.split("\n")
            total_lines = len(lines)

            if should_read_entire_file:
                return {
                    "success": True,
                    "content": content,
                    "target_file": target_file,
                    "total_lines": total_lines,
                    "explanation": explanation,
                }
            else:
                # Handle line range reading
                start_idx = max(0, start_line_one_indexed - 1)
                end_idx = (
                    min(total_lines, end_line_one_indexed_inclusive)
                    if end_line_one_indexed_inclusive != -1
                    else total_lines
                )

                selected_lines = lines[start_idx:end_idx]
                selected_content = "\n".join(selected_lines)

                return {
                    "success": True,
                    "content": selected_content,
                    "target_file": target_file,
                    "start_line": start_line_one_indexed,
                    "end_line": end_idx,
                    "total_lines": total_lines,
                    "lines_before": start_idx,
                    "lines_after": total_lines - end_idx,
                    "explanation": explanation,
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"Error reading file '{target_file}': {str(e)}",
            }

    def run_terminal_cmd(
        self, command: str, explanation: str = "", is_background: bool = False
    ) -> dict:
        """Execute a terminal command"""
        try:
            if self.container:
                cmd_list = ["sh", "-c", command]
                result = run_command_in_container(self.container, cmd_list)
                return {
                    "success": result["success"],
                    "output": result.get("output", ""),
                    "error": result.get("error", ""),
                    "command": command,
                    "explanation": explanation,
                    "is_background": is_background,
                }
            else:
                # Local execution
                if is_background:
                    # For background processes, use Popen
                    process = subprocess.Popen(
                        command,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        cwd=str(self.base_path),
                    )
                    return {
                        "success": True,
                        "message": f"Background process started with PID {process.pid}",
                        "pid": process.pid,
                        "command": command,
                        "explanation": explanation,
                        "is_background": True,
                    }
                else:
                    # Synchronous execution
                    result = subprocess.run(
                        command,
                        shell=True,
                        capture_output=True,
                        text=True,
                        cwd=str(self.base_path),
                        timeout=30,  # 30 second timeout
                    )
                    return {
                        "success": result.returncode == 0,
                        "output": result.stdout,
                        "error": result.stderr,
                        "return_code": result.returncode,
                        "command": command,
                        "explanation": explanation,
                        "is_background": False,
                    }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Command timed out after 30 seconds: {command}",
                "command": command,
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error executing command '{command}': {str(e)}",
                "command": command,
            }

    def list_dir(self, relative_workspace_path: str, explanation: str = "") -> dict:
        """List directory contents"""
        try:
            if self.container:
                # Normalize to a relative path within base_path
                safe_rel_path = relative_workspace_path.lstrip("/") or "."
                # Use shell to capture stderr as well (2>&1)
                list_cmd = f"ls -la {self.base_path / safe_rel_path} 2>&1"
                result = run_command_in_container(
                    container=self.container,
                    command=["sh", "-c", list_cmd],
                )
                if result["success"]:
                    return {
                        "success": True,
                        "contents": result["output"],
                        "path": safe_rel_path,
                        "explanation": explanation,
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Failed to list directory: {result.get('output') or 'Unknown error'}",
                        "path": safe_rel_path,
                        "command": list_cmd,
                    }
            else:
                # Local file system
                safe_rel_path = relative_workspace_path.lstrip("/") or "."
                full_path = self.base_path / safe_rel_path
                if full_path.exists() and full_path.is_dir():
                    contents = []
                    for item in full_path.iterdir():
                        stat_info = item.stat()
                        contents.append(
                            {
                                "name": item.name,
                                "type": "directory" if item.is_dir() else "file",
                                "size": stat_info.st_size if item.is_file() else None,
                                "permissions": oct(stat_info.st_mode)[-3:],
                            }
                        )
                    return {
                        "success": True,
                        "contents": contents,
                        "path": safe_rel_path,
                        "explanation": explanation,
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Directory '{safe_rel_path}' does not exist or is not a directory",
                    }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error listing directory '{relative_workspace_path}': {str(e)}",
            }

    def grep_search(
        self,
        query: str,
        explanation: str = "",
        case_sensitive: bool = False,
        include_pattern: str = None,
        exclude_pattern: str = None,
    ) -> dict:
        """Search for text patterns using grep/ripgrep"""
        try:
            # Build search command
            if self.container:

                # Check if ripgrep is available inside the container, fallback to grep. Commenting out for now.
                # rg_check = run_command_in_container(self.container, ["which", "rg"])
                # cmd = ["rg"] if rg_check["success"] else ["grep", "-r"]

                if not case_sensitive and cmd[0] == "rg":
                    cmd.append("-i")
                elif not case_sensitive and cmd[0] == "grep":
                    cmd.append("-i")

                if include_pattern:
                    if cmd[0] == "rg":
                        cmd.extend(["-g", include_pattern])
                    else:
                        cmd.extend(["--include", include_pattern])

                if exclude_pattern:
                    if cmd[0] == "rg":
                        cmd.extend(["-g", f"!{exclude_pattern}"])
                    else:
                        cmd.extend(["--exclude", exclude_pattern])

                cmd.extend([query, str(self.base_path)])

                result = run_command_in_container(self.container, cmd)
                return {
                    "success": result["success"],
                    "matches": result.get("output", "").split("\n")
                    if result["success"]
                    else [],
                    "query": query,
                    "explanation": explanation,
                }
            else:
                # Local search
                cmd = ["rg"] if shutil.which("rg") else ["grep", "-r"]

                if not case_sensitive:
                    cmd.append("-i")

                if include_pattern and cmd[0] == "rg":
                    cmd.extend(["-g", include_pattern])
                elif include_pattern and cmd[0] == "grep":
                    cmd.extend(["--include", include_pattern])

                if exclude_pattern and cmd[0] == "rg":
                    cmd.extend(["-g", f"!{exclude_pattern}"])
                elif exclude_pattern and cmd[0] == "grep":
                    cmd.extend(["--exclude", exclude_pattern])

                cmd.extend([query, str(self.base_path)])

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                matches = result.stdout.split("\n") if result.stdout else []

                return {
                    "success": result.returncode == 0,
                    "matches": [m for m in matches if m.strip()],
                    "query": query,
                    "explanation": explanation,
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"Grep search failed: {str(e)}",
                "query": query,
            }

    def edit_file(
        self,
        target_file: str,
        instructions: str,
        edit_type: str = "line_edits",
        line_edits: list = None,
    ) -> dict:
        """
        Edit a file using structured line-based editing

        Args:
            target_file: Path to the file to edit
            instructions: Description of the edit
            edit_type: Must be 'line_edits' (only supported type)
            line_edits: List of line-based edits
        """
        try:
            if edit_type != "line_edits":
                return {
                    "success": False,
                    "error": f"Only 'line_edits' edit_type is supported, got: {edit_type}",
                }

            if self.container:
                # Handle container-based file operations
                container_file_path = str(self.base_path / target_file)

                # Check if file exists in container, create if needed
                check_result = run_command_in_container(
                    self.container, ["test", "-f", container_file_path]
                )

                if not check_result["success"]:
                    # File doesn't exist, create parent directories and file
                    parent_dir = str(Path(container_file_path).parent)
                    run_command_in_container(
                        self.container, ["mkdir", "-p", parent_dir]
                    )
                    run_command_in_container(
                        self.container, ["touch", container_file_path]
                    )

                return self._apply_line_edits(
                    container_file_path, line_edits, instructions
                )
            else:
                # Local filesystem operations
                file_path = self.base_path / target_file

                # Check if file exists, create parent directories if needed
                if not file_path.exists():
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    # Create empty file if it doesn't exist
                    file_path.touch()

                return self._apply_line_edits(file_path, line_edits, instructions)

        except Exception as e:
            return {
                "success": False,
                "error": f"File edit failed: {str(e)}",
                "target_file": target_file,
            }


    def _apply_line_edits(self, file_path, line_edits: list, instructions: str) -> dict:
        """Apply a list of line-based edits to a file"""
        try:
            if not line_edits:
                return {
                    "success": False,
                    "error": "line_edits is required for line_edits edit type",
                }

            # Read current file content
            if self.container:
                # Container-based file reading
                read_result = run_command_in_container(
                    self.container, ["cat", file_path]
                )
                if read_result["success"]:
                    content = read_result["output"]
                    lines = content.splitlines(keepends=True) if content else []
                else:
                    lines = []  # File doesn't exist or can't be read
            else:
                # Local file reading
                file_path_obj = (
                    file_path if isinstance(file_path, Path) else Path(file_path)
                )
                if file_path_obj.exists():
                    with open(file_path_obj, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                else:
                    lines = []

            # Process edits in a way that handles line number shifts correctly
            # Sort by line number, but handle different operations appropriately
            changes_made = []

            # Keep a copy of original content to detect no-op edits
            original_content = "".join(lines)

            # Group edits by type for better handling
            deletes = [e for e in line_edits if e.get("type") == "delete"]
            replaces = [e for e in line_edits if e.get("type") == "replace"]
            inserts = [e for e in line_edits if e.get("type") == "insert"]

            # Apply deletes first (in reverse line order)
            for edit in sorted(
                deletes, key=lambda x: x.get("line_number", 0), reverse=True
            ):
                line_number = edit.get("line_number", 1)
                line_idx = max(0, line_number - 1)

                if line_idx < len(lines):
                    deleted_content = lines[line_idx].rstrip("\n")
                    del lines[line_idx]
                    changes_made.append(
                        f"Line {line_number}: Deleted '{deleted_content[:50]}...'"
                    )
                else:
                    changes_made.append(
                        f"Line {line_number}: Could not delete (line doesn't exist)"
                    )

            # Apply replaces (in reverse line order)
            for edit in sorted(
                replaces, key=lambda x: x.get("line_number", 0), reverse=True
            ):
                line_number = edit.get("line_number", 1)
                content = edit.get("content", "")
                line_idx = max(0, line_number - 1)

                if line_idx < len(lines):
                    old_content = lines[line_idx].rstrip("\n")
                    lines[line_idx] = content + "\n"
                    changes_made.append(
                        f"Line {line_number}: Replaced '{old_content[:50]}...' with '{content[:50]}...'"
                    )
                else:
                    # Line doesn't exist, extend the file to that line
                    while len(lines) < line_number:
                        lines.append("\n")
                    lines[line_number - 1] = content + "\n"
                    changes_made.append(
                        f"Line {line_number}: Added '{content[:50]}...' (extended file)"
                    )

            # Apply inserts (in reverse line order to avoid shifting issues)
            for edit in sorted(
                inserts, key=lambda x: x.get("line_number", 0), reverse=True
            ):
                line_number = edit.get("line_number", 1)
                content = edit.get("content", "")
                line_idx = max(0, line_number - 1)

                # Insert at the specified position
                if line_idx <= len(lines):
                    lines.insert(line_idx, content + "\n")
                    changes_made.append(
                        f"Line {line_number}: Inserted '{content[:50]}...'"
                    )
                else:
                    # Beyond end of file, just append
                    lines.append(content + "\n")
                    changes_made.append(
                        f"Line {line_number}: Appended '{content[:50]}...' (beyond file end)"
                    )

            # Validate Python syntax if this is a Python file
            content = "".join(lines)

            # Detect no-op edit (no effective change in file text)
            if content == original_content:
                return {
                    "success": False,
                    "error": "No effective change (new content identical to existing file)",
                    "target_file": str(Path(file_path).relative_to(Path(self.base_path))) if hasattr(self, 'base_path') else str(file_path),
                    "instructions": instructions,
                    "total_edits": len(line_edits),
                }
            file_ext = Path(file_path).suffix.lower()

            if file_ext == '.py':
                try:
                    import ast
                    ast.parse(content)
                    print(f"HARNESS: Python syntax validation passed for {file_path}")
                except SyntaxError as e:
                    error_msg = f"Python syntax error in {file_path}: {e.msg} at line {e.lineno}"
                    print(f"HARNESS: {error_msg}")
                    print(f"HARNESS: Changes that would have been applied: {changes_made}")
                    return {
                        "success": False,
                        "error": error_msg,
                        "target_file": str(Path(file_path).relative_to(Path(self.base_path))) if hasattr(self, 'base_path') else str(file_path),
                        "syntax_error": True,
                        "syntax_line": e.lineno,
                        "syntax_msg": e.msg,
                        "attempted_changes": changes_made
                    }
                except Exception as e:
                    print(f"HARNESS: Could not validate Python syntax: {e}")

            # Write the modified content back to file
            if self.container:
                # Enhanced container-based file writing with better error handling
                print(f"HARNESS: Writing {len(content)} characters to {file_path}")

                # Method 1: Try python-based writing first (most reliable)
                import base64
                encoded_content = base64.b64encode(content.encode('utf-8')).decode('ascii')
                python_write_cmd = [
                    "python3", "-c",
                    f"import base64; "
                    f"content = base64.b64decode('{encoded_content}').decode('utf-8'); "
                    f"open('{file_path}', 'w', encoding='utf-8').write(content)"
                ]
                write_result = run_command_in_container(self.container, python_write_cmd)

                if not write_result["success"]:
                    print(f"HARNESS: Python write failed, trying cat method...")
                    # Fallback: Use cat with proper content handling
                    # Create a temporary file with safe content then copy it
                    import tempfile
                    import shlex

                    # Write to a temporary file path that's safe
                    temp_path = f"/tmp/edit_content_{hash(content) % 10000}.tmp"
                    escaped_content = content.replace("'", "'\"'\"'")  # Escape single quotes properly
                    cat_cmd = ["sh", "-c", f"cat > {temp_path} << 'EDIT_EOF'\n{escaped_content}\nEDIT_EOF\n && mv {temp_path} {file_path}"]

                    write_result = run_command_in_container(self.container, cat_cmd)

                if not write_result["success"]:
                    print(f"HARNESS: Both write methods failed!")
                    print(f"HARNESS: Write error: {write_result.get('error', 'Unknown error')}")
                    return {
                        "success": False,
                        "error": f"Failed to write file to container: {write_result.get('error', 'Unknown error')}",
                        "target_file": str(
                            Path(file_path).relative_to(Path(self.base_path))
                        ),
                    }

                print(f"HARNESS: File write succeeded, verifying content...")

                # Enhanced verification with detailed comparison
                verify_read = run_command_in_container(self.container, ["cat", file_path])
                if not verify_read.get("success"):
                    print(f"HARNESS: Verification read failed: {verify_read.get('error', 'Unknown error')}")
                    return {
                        "success": False,
                        "error": f"Failed to read file back for verification: {verify_read.get('error', 'Unknown error')}",
                        "target_file": str(
                            Path(file_path).relative_to(Path(self.base_path))
                        ),
                    }

                # Normalize for trailing newline and CRLF vs LF differences
                def _norm(s: str) -> str:
                    return s.replace("\r\n", "\n").rstrip("\n")

                written_content = verify_read.get("output", "")
                expected_normalized = _norm(content)
                actual_normalized = _norm(written_content)

                if actual_normalized != expected_normalized:
                    print(f"HARNESS: Content verification failed!")
                    print(f"HARNESS: Expected length: {len(expected_normalized)}")
                    print(f"HARNESS: Actual length: {len(actual_normalized)}")
                    print(f"HARNESS: Expected preview: {expected_normalized[:200]}...")
                    print(f"HARNESS: Actual preview: {actual_normalized[:200]}...")
                    return {
                        "success": False,
                        "error": "Write verification failed (content mismatch after write)",
                        "target_file": str(
                            Path(file_path).relative_to(Path(self.base_path))
                        ),
                        "expected_length": len(expected_normalized),
                        "actual_length": len(actual_normalized),
                    }

                print(f"HARNESS: Content verification passed!")

                # Stage changes to make them visible to diff tools
                print(f"HARNESS: Staging changes with git...")
                git_add_result = run_command_in_container(
                    self.container,
                    ["git", "-C", "/app", "add", "-A"],
                )

                if not git_add_result.get("success"):
                    print(f"HARNESS: Git add failed: {git_add_result.get('error', 'Unknown error')}")

                # Check git status for debugging
                git_status_result = run_command_in_container(
                    self.container,
                    ["git", "-C", "/app", "status", "--porcelain"],
                )
                if git_status_result.get("success"):
                    print(f"HARNESS: Git status after edit: {git_status_result.get('output', '').strip()}")

                # Log staged changes for visibility
                staged_names = run_command_in_container(
                    self.container,
                    ["git", "-C", "/app", "diff", "--cached", "--name-only"],
                )
                if staged_names.get("success"):
                    print(f"HARNESS: Staged files after edit: {staged_names.get('output', '').strip()}")
                else:
                    print(f"HARNESS: Failed to check staged files: {staged_names.get('error', 'Unknown error')}")

                file_name = Path(file_path).name
                target_file = str(Path(file_path).relative_to(Path(self.base_path)))
            else:
                # Local file writing
                file_path_obj = (
                    file_path if isinstance(file_path, Path) else Path(file_path)
                )
                with open(file_path_obj, "w", encoding="utf-8") as f:
                    f.writelines(lines)

                # Verify persistence locally
                with open(file_path_obj, "r", encoding="utf-8") as f:
                    read_back = f.read()
                    def _norm_local(s: str) -> str:
                        return s.replace("\r\n", "\n").rstrip("\n")
                    if _norm_local(read_back) != _norm_local(content):
                        return {
                            "success": False,
                            "error": "Write verification failed (content mismatch after write)",
                            "target_file": str(file_path_obj.relative_to(self.base_path)),
                        }

                file_name = file_path_obj.name
                target_file = str(file_path_obj.relative_to(self.base_path))

            return {
                "success": True,
                "message": f"Successfully applied {len(line_edits)} line edits to {file_name}",
                "target_file": target_file,
                "instructions": instructions,
                "changes_made": changes_made,
                "total_edits": len(line_edits),
            }

        except Exception as e:
            target_file = (
                str(Path(file_path).relative_to(Path(self.base_path)))
                if isinstance(file_path, str)
                else str(file_path.relative_to(self.base_path))
            )
            return {
                "success": False,
                "error": f"Error applying line edits: {str(e)}",
                "target_file": target_file,
            }

    def search_replace(self, file_path: str, old_string: str, new_string: str) -> dict:
        """Search and replace text in a file"""
        try:
            if self.container:
                # Read file first
                read_result = run_command_in_container(
                    self.container, ["cat", str(self.base_path / file_path)]
                )
                if not read_result["success"]:
                    return {
                        "success": False,
                        "error": f"Failed to read file: {read_result.get('error', 'Unknown error')}",
                    }

                content = read_result["output"]

                # Perform replacement
                if old_string in content:
                    new_content = content.replace(
                        old_string, new_string, 1
                    )  # Replace only first occurrence

                    # Write back to file
                    escaped_content = new_content.replace('"', '\\"').replace(
                        "$", "\\$"
                    )
                    write_result = run_command_in_container(
                        self.container,
                        [
                            "sh",
                            "-c",
                            f'echo "{escaped_content}" > {self.base_path / file_path}',
                        ],
                    )

                    return {
                        "success": write_result["success"],
                        "message": f"Successfully replaced text in {file_path}",
                        "file_path": file_path,
                        "old_string": old_string[:100] + "..."
                        if len(old_string) > 100
                        else old_string,
                        "new_string": new_string[:100] + "..."
                        if len(new_string) > 100
                        else new_string,
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Old string not found in file {file_path}",
                        "file_path": file_path,
                    }
            else:
                # Local file system
                full_path = self.base_path / file_path
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()

                if old_string in content:
                    new_content = content.replace(
                        old_string, new_string, 1
                    )  # Replace only first occurrence

                    with open(full_path, "w", encoding="utf-8") as f:
                        f.write(new_content)

                    return {
                        "success": True,
                        "message": f"Successfully replaced text in {file_path}",
                        "file_path": file_path,
                        "old_string": old_string[:100] + "..."
                        if len(old_string) > 100
                        else old_string,
                        "new_string": new_string[:100] + "..."
                        if len(new_string) > 100
                        else new_string,
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Old string not found in file {file_path}",
                        "file_path": file_path,
                    }

        except Exception as e:
            return {
                "success": False,
                "error": f"Search and replace failed: {str(e)}",
                "file_path": file_path,
            }

    def file_search(self, query: str, explanation: str = "") -> dict:
        """Search for files by name using fuzzy matching"""
        try:
            results = []
            if self.container:
                # Use find command to search for files
                cmd = ["find", str(self.base_path), "-name", f"*{query}*", "-type", "f"]
                result = run_command_in_container(self.container, cmd)
                if result["success"]:
                    files = [
                        f.strip() for f in result["output"].split("\n") if f.strip()
                    ]
                    for file in files[:10]:  # Limit to 10 results
                        rel_path = (
                            Path(file).relative_to(self.base_path)
                            if str(self.base_path) in file
                            else file
                        )
                        results.append(str(rel_path))
            else:
                # Local fuzzy search
                search_path = self.base_path
                for file in search_path.rglob("*"):
                    if file.is_file() and query.lower() in file.name.lower():
                        results.append(str(file.relative_to(self.base_path)))
                        if len(results) >= 10:  # Limit results
                            break

            return {
                "success": True,
                "results": results,
                "query": query,
                "explanation": explanation,
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"File search failed: {str(e)}",
                "query": query,
            }

    def delete_file(self, target_file: str, explanation: str = "") -> dict:
        """Delete a file"""
        try:
            if self.container:
                result = run_command_in_container(
                    self.container, ["rm", str(self.base_path / target_file)]
                )
                return {
                    "success": result["success"],
                    "message": f"File {target_file} deleted"
                    if result["success"]
                    else "Failed to delete file",
                    "target_file": target_file,
                    "explanation": explanation,
                    "error": result.get("error") if not result["success"] else None,
                }
            else:
                # Local file deletion
                full_path = self.base_path / target_file
                if full_path.exists():
                    full_path.unlink()
                    return {
                        "success": True,
                        "message": f"File {target_file} deleted successfully",
                        "target_file": target_file,
                        "explanation": explanation,
                    }
                else:
                    return {
                        "success": False,
                        "error": f"File {target_file} does not exist",
                        "target_file": target_file,
                    }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to delete file {target_file}: {str(e)}",
                "target_file": target_file,
            }

    def web_search(self, search_term: str, explanation: str = "") -> dict:
        """Search the web using DuckDuckGo instant answers API"""
        try:
            import urllib.request
            import urllib.parse
            import json

            # Use DuckDuckGo instant answers API (no API key required)
            encoded_term = urllib.parse.quote_plus(search_term)
            url = f"https://api.duckduckgo.com/?q={encoded_term}&format=json&no_html=1&skip_disambig=1"

            try:
                with urllib.request.urlopen(url, timeout=10) as response:
                    data = json.loads(response.read().decode())

                    results = []

                    # Add abstract if available
                    if data.get("Abstract"):
                        results.append({
                            "title": data.get("AbstractSource", "Summary"),
                            "snippet": data["Abstract"],
                            "url": data.get("AbstractURL", ""),
                            "type": "summary"
                        })

                    # Add instant answer if available
                    if data.get("Answer"):
                        results.append({
                            "title": "Instant Answer",
                            "snippet": data["Answer"],
                            "url": "",
                            "type": "answer"
                        })

                    # Add related topics
                    for topic in data.get("RelatedTopics", [])[:3]:
                        if isinstance(topic, dict) and topic.get("Text"):
                            results.append({
                                "title": topic.get("FirstURL", "").split("/")[-1] or "Related",
                                "snippet": topic["Text"],
                                "url": topic.get("FirstURL", ""),
                                "type": "related"
                            })

                    return {
                        "success": True,
                        "search_term": search_term,
                        "results": results,
                        "explanation": explanation,
                        "note": "Results from DuckDuckGo instant answers API"
                    }

            except Exception as e:
                # Fallback: return a simulated web search result
                return {
                    "success": True,
                    "search_term": search_term,
                    "results": [{
                        "title": f"Search results for: {search_term}",
                        "snippet": f"Web search functionality simulated. In a real implementation, this would query search engines for '{search_term}' and return relevant web results.",
                        "url": f"https://duckduckgo.com/?q={encoded_term}",
                        "type": "fallback"
                    }],
                    "explanation": explanation,
                    "note": f"Fallback result due to network error: {str(e)}"
                }

        except ImportError:
            return {
                "success": False,
                "error": "Required modules not available for web search",
                "search_term": search_term,
                "explanation": explanation
            }

    def create_diagram(self, content: str) -> dict:
        """Create and validate a Mermaid diagram"""
        try:
            # Basic validation of Mermaid syntax
            valid_diagram_types = [
                "graph", "flowchart", "sequenceDiagram", "classDiagram",
                "stateDiagram", "erDiagram", "journey", "gitgraph",
                "pie", "quadrantChart", "timeline", "mindmap"
            ]

            content_lower = content.lower().strip()
            is_valid = any(content_lower.startswith(diagram_type.lower()) for diagram_type in valid_diagram_types)

            if not is_valid:
                # Try to detect if it's a simple graph syntax
                if any(symbol in content for symbol in ["-->", "---", "->", "--"]):
                    # Likely a graph, prepend with graph directive
                    content = f"graph TD\n{content}"
                    is_valid = True

            # Create a diagram file for visualization
            diagram_filename = "generated_diagram.mmd"
            if self.container:
                # Save to container
                result = run_command_in_container(
                    container=self.container,
                    command=["sh", "-c", f'echo "{content}" > {self.base_path / diagram_filename}']
                )
                file_created = result["success"]
            else:
                # Save locally
                try:
                    with open(self.base_path / diagram_filename, 'w', encoding='utf-8') as f:
                        f.write(content)
                    file_created = True
                except Exception:
                    file_created = False

            return {
                "success": True,
                "message": "Mermaid diagram created and validated",
                "content": content,
                "is_valid_syntax": is_valid,
                "file_created": file_created,
                "filename": diagram_filename,
                "note": f"Mermaid diagram {'validated and ' if is_valid else 'created (syntax may need review) and '}saved to {diagram_filename}",
                "preview_url": f"https://mermaid.live/edit#{content.replace(' ', '%20').replace(chr(10), '%0A')}"
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to create diagram: {str(e)}",
                "content": content
            }

    def edit_notebook(
        self,
        target_notebook: str,
        cell_idx: int,
        is_new_cell: bool,
        cell_language: str,
        old_string: str,
        new_string: str,
    ) -> dict:
        """Edit a Jupyter notebook cell with full .ipynb support"""
        try:
            import json

            # Read the notebook file
            if self.container:
                result = run_command_in_container(
                    container=self.container,
                    command=["cat", str(self.base_path / target_notebook)]
                )
                if not result["success"]:
                    return {
                        "success": False,
                        "error": f"Failed to read notebook: {result.get('error', 'Unknown error')}"
                    }
                notebook_content = result["output"]
            else:
                try:
                    with open(self.base_path / target_notebook, 'r', encoding='utf-8') as f:
                        notebook_content = f.read()
                except FileNotFoundError:
                    # Create new notebook if it doesn't exist
                    notebook_content = json.dumps({
                        "cells": [],
                        "metadata": {},
                        "nbformat": 4,
                        "nbformat_minor": 4
                    })

            # Parse the notebook JSON
            try:
                notebook = json.loads(notebook_content)
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "error": f"Invalid notebook JSON: {str(e)}"
                }

            # Ensure cells array exists
            if "cells" not in notebook:
                notebook["cells"] = []

            # Determine cell type
            cell_type = "code" if cell_language == "python" else "markdown"
            if cell_language in ["python", "javascript", "typescript", "r", "sql", "shell"]:
                cell_type = "code"
            elif cell_language in ["markdown", "raw"]:
                cell_type = cell_language

            if is_new_cell:
                # Create new cell
                new_cell = {
                    "cell_type": cell_type,
                    "metadata": {},
                    "source": new_string.split('\n') if new_string else [""]
                }

                if cell_type == "code":
                    new_cell["execution_count"] = None
                    new_cell["outputs"] = []

                # Insert at specified index
                if cell_idx <= len(notebook["cells"]):
                    notebook["cells"].insert(cell_idx, new_cell)
                else:
                    notebook["cells"].append(new_cell)

                action = f"Created new {cell_type} cell at index {cell_idx}"
            else:
                # Edit existing cell
                if cell_idx >= len(notebook["cells"]):
                    return {
                        "success": False,
                        "error": f"Cell index {cell_idx} out of range (notebook has {len(notebook['cells'])} cells)"
                    }

                cell = notebook["cells"][cell_idx]
                current_source = "\n".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]

                if old_string in current_source:
                    new_source = current_source.replace(old_string, new_string)
                    cell["source"] = new_source.split('\n')
                    action = f"Edited cell {cell_idx}"
                else:
                    return {
                        "success": False,
                        "error": f"Old string not found in cell {cell_idx}"
                    }

            # Write back the notebook
            updated_content = json.dumps(notebook, indent=2)

            if self.container:
                # Escape content for shell
                escaped_content = updated_content.replace('"', '\\"').replace('$', '\\$').replace('`', '\\`')
                result = run_command_in_container(
                    container=self.container,
                    command=["sh", "-c", f'echo "{escaped_content}" > {self.base_path / target_notebook}']
                )
                write_success = result["success"]
            else:
                try:
                    with open(self.base_path / target_notebook, 'w', encoding='utf-8') as f:
                        f.write(updated_content)
                    write_success = True
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Failed to write notebook: {str(e)}"
                    }

            if write_success:
                return {
                    "success": True,
                    "message": action,
                    "target_notebook": target_notebook,
                    "cell_idx": cell_idx,
                    "cell_type": cell_type,
                    "is_new_cell": is_new_cell,
                    "total_cells": len(notebook["cells"])
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to write notebook file"
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"Error editing notebook: {str(e)}",
                "target_notebook": target_notebook
            }

    def write_file(self, file_path: str, content: str) -> dict:
        """Write content to a file - kept for backward compatibility"""
        try:
            if self.container:
                # Use echo to write content to file in container
                escaped_content = content.replace('"', '\\"').replace("$", "\\$")
                result = run_command_in_container(
                    container=self.container,
                    command=[
                        "sh",
                        "-c",
                        f'echo "{escaped_content}" > {self.base_path / file_path}',
                    ],
                )
                return {
                    "success": result["success"],
                    "message": f"Successfully wrote to {file_path}"
                    if result["success"]
                    else "Failed to write file",
                    "file_path": file_path,
                    "error": result.get("error") if not result["success"] else None,
                }
            else:
                # Local file system
                full_path = self.base_path / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(content)
                return {
                    "success": True,
                    "message": f"Successfully wrote to {file_path}",
                    "file_path": file_path,
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error writing file '{file_path}': {str(e)}",
            }

    def list_files(self, directory: str = ".") -> dict:
        """List files in a directory - kept for backward compatibility"""
        try:
            if self.container:
                result = run_command_in_container(
                    container=self.container,
                    command=[
                        "find",
                        str(self.base_path / directory),
                        "-type",
                        "f",
                        "-name",
                        "*",
                    ],
                )
                if result["success"]:
                    files = [
                        line.strip()
                        for line in result["output"].split("\n")
                        if line.strip()
                    ]
                    relative_files = []
                    for file in files:
                        try:
                            rel_path = Path(file).relative_to(self.base_path)
                            relative_files.append(str(rel_path))
                        except ValueError:
                            relative_files.append(file)
                    return {
                        "success": True,
                        "files": relative_files,
                        "directory": directory,
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Failed to list files: {result.get('error', 'Unknown error')}",
                    }
            else:
                # Local file system
                full_path = self.base_path / directory
                if full_path.exists() and full_path.is_dir():
                    files = [
                        str(p.relative_to(self.base_path))
                        for p in full_path.rglob("*")
                        if p.is_file()
                    ]
                    return {"success": True, "files": files, "directory": directory}
                else:
                    return {
                        "success": False,
                        "error": f"Directory '{directory}' does not exist",
                    }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error listing files in '{directory}': {str(e)}",
            }

    def create_directory(self, directory_path: str) -> dict:
        """Create a directory - kept for backward compatibility"""
        try:
            if self.container:
                result = run_command_in_container(
                    container=self.container,
                    command=["mkdir", "-p", str(self.base_path / directory_path)],
                )
                return {
                    "success": result["success"],
                    "message": f"Successfully created directory {directory_path}"
                    if result["success"]
                    else "Failed to create directory",
                    "directory_path": directory_path,
                    "error": result.get("error") if not result["success"] else None,
                }
            else:
                # Local file system
                full_path = self.base_path / directory_path
                full_path.mkdir(parents=True, exist_ok=True)
                return {
                    "success": True,
                    "message": f"Successfully created directory {directory_path}",
                    "directory_path": directory_path,
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error creating directory '{directory_path}': {str(e)}",
            }

    def api_call(self, method: str, url: str, headers: dict = None, body: dict = None, explanation: str = "") -> dict:
        """Make HTTP API calls to test backend functionality"""
        try:
            import urllib.request
            import urllib.parse
            import json

            # Handle relative URLs
            if not url.startswith("http"):
                url = f"{self.mern_config['api_base_url']}{url if url.startswith('/') else '/' + url}"

            # Prepare request
            req_headers = headers or {}
            req_headers.setdefault("Content-Type", "application/json")

            request_data = None
            if body and method in ["POST", "PUT", "PATCH"]:
                request_data = json.dumps(body).encode('utf-8')

            req = urllib.request.Request(url, data=request_data, headers=req_headers, method=method)

            try:
                with urllib.request.urlopen(req, timeout=10) as response:
                    response_data = response.read().decode('utf-8')
                    try:
                        json_data = json.loads(response_data)
                    except json.JSONDecodeError:
                        json_data = response_data

                    return {
                        "success": True,
                        "status_code": response.status,
                        "data": json_data,
                        "url": url,
                        "method": method,
                        "explanation": explanation
                    }
            except urllib.error.HTTPError as e:
                error_data = e.read().decode('utf-8')
                try:
                    error_json = json.loads(error_data)
                except json.JSONDecodeError:
                    error_json = error_data

                return {
                    "success": False,
                    "status_code": e.code,
                    "error": error_json,
                    "url": url,
                    "method": method,
                    "explanation": explanation
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"API call failed: {str(e)}",
                "url": url,
                "method": method,
                "explanation": explanation
            }

    def database_query(self, query_type: str, collection: str, query: dict = None, data: dict = None, explanation: str = "") -> dict:
        """Execute MongoDB queries to test data persistence"""
        try:
            # Try using pymongo if available
            try:
                from pymongo import MongoClient

                client = MongoClient(self.mern_config['mongo_uri'])
                db = client.get_default_database()
                coll = db[collection]

                result = None
                if query_type == "find":
                    result = list(coll.find(query or {}))
                elif query_type == "insert":
                    result = coll.insert_one(data or {})
                    result = {"inserted_id": str(result.inserted_id)}
                elif query_type == "update":
                    result = coll.update_many(query or {}, {"$set": data or {}})
                    result = {"matched_count": result.matched_count, "modified_count": result.modified_count}
                elif query_type == "delete":
                    result = coll.delete_many(query or {})
                    result = {"deleted_count": result.deleted_count}
                elif query_type == "aggregate":
                    result = list(coll.aggregate(query or []))

                client.close()
                return {
                    "success": True,
                    "result": result,
                    "collection": collection,
                    "query_type": query_type,
                    "explanation": explanation
                }

            except ImportError:
                # Fallback: use mongosh command if pymongo not available
                if self.container:
                    mongo_cmd = f"mongosh {self.mern_config['mongo_uri']} --eval 'db.{collection}.{query_type}({json.dumps(query or {})})''"
                    result = run_command_in_container(self.container, ["sh", "-c", mongo_cmd])
                    return {
                        "success": result["success"],
                        "result": result.get("output", ""),
                        "collection": collection,
                        "query_type": query_type,
                        "explanation": explanation
                    }
                else:
                    return {
                        "success": False,
                        "error": "PyMongo not available and not in container environment",
                        "explanation": explanation
                    }

        except Exception as e:
            return {
                "success": False,
                "error": f"Database query failed: {str(e)}",
                "collection": collection,
                "query_type": query_type,
                "explanation": explanation
            }

    def websocket_test(self, event_name: str, event_data: dict = None, expected_response: str = "", timeout: int = 10, explanation: str = "") -> dict:
        """Test WebSocket/Socket.IO functionality"""
        try:
            # Try using socketio client if available
            try:
                import socketio

                sio = socketio.SimpleClient()
                sio.connect(self.mern_config['websocket_url'])

                # Send event
                sio.emit(event_name, event_data or {})

                # Wait for response if expected
                if expected_response:
                    try:
                        response = sio.receive(timeout=timeout)
                        sio.disconnect()
                        return {
                            "success": True,
                            "event_sent": event_name,
                            "response_received": response,
                            "explanation": explanation
                        }
                    except Exception as e:
                        sio.disconnect()
                        return {
                            "success": False,
                            "error": f"No response received: {str(e)}",
                            "event_sent": event_name,
                            "explanation": explanation
                        }
                else:
                    sio.disconnect()
                    return {
                        "success": True,
                        "event_sent": event_name,
                        "message": "Event sent successfully (no response expected)",
                        "explanation": explanation
                    }

            except ImportError:
                # Fallback: simulate websocket test
                return {
                    "success": True,
                    "event_sent": event_name,
                    "message": "WebSocket test simulated (socketio client not available)",
                    "note": "Install python-socketio for real WebSocket testing",
                    "explanation": explanation
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"WebSocket test failed: {str(e)}",
                "event_name": event_name,
                "explanation": explanation
            }

    def ui_test(self, action: str, selector: str = "", text: str = "", url: str = "", timeout: int = 10, explanation: str = "") -> dict:
        """Test frontend UI functionality using browser automation"""
        try:
            # Try using selenium if available
            try:
                from selenium import webdriver
                from selenium.webdriver.common.by import By
                from selenium.webdriver.support.ui import WebDriverWait
                from selenium.webdriver.support import expected_conditions as EC
                from selenium.webdriver.chrome.options import Options

                # Setup headless Chrome
                chrome_options = Options()
                chrome_options.add_argument("--headless")
                chrome_options.add_argument("--no-sandbox")
                chrome_options.add_argument("--disable-dev-shm-usage")

                driver = webdriver.Chrome(options=chrome_options)
                driver.set_page_load_timeout(timeout)

                try:
                    if action == "navigate":
                        target_url = url or self.mern_config['frontend_url']
                        driver.get(target_url)
                        return {
                            "success": True,
                            "action": action,
                            "url": target_url,
                            "page_title": driver.title,
                            "explanation": explanation
                        }

                    elif action == "screenshot":
                        screenshot_path = "/tmp/ui_screenshot.png"
                        driver.save_screenshot(screenshot_path)
                        return {
                            "success": True,
                            "action": action,
                            "screenshot_path": screenshot_path,
                            "page_title": driver.title,
                            "explanation": explanation
                        }

                    elif action == "click":
                        element = WebDriverWait(driver, timeout).until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                        )
                        element.click()
                        return {
                            "success": True,
                            "action": action,
                            "selector": selector,
                            "explanation": explanation
                        }

                    elif action == "type":
                        element = WebDriverWait(driver, timeout).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                        )
                        element.clear()
                        element.send_keys(text)
                        return {
                            "success": True,
                            "action": action,
                            "selector": selector,
                            "text": text,
                            "explanation": explanation
                        }

                    elif action == "get_text":
                        element = WebDriverWait(driver, timeout).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                        )
                        element_text = element.text
                        return {
                            "success": True,
                            "action": action,
                            "selector": selector,
                            "text": element_text,
                            "explanation": explanation
                        }

                    elif action == "wait_for_element":
                        element = WebDriverWait(driver, timeout).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                        )
                        return {
                            "success": True,
                            "action": action,
                            "selector": selector,
                            "message": "Element found",
                            "explanation": explanation
                        }

                    else:
                        return {
                            "success": False,
                            "error": f"Unknown UI action: {action}",
                            "explanation": explanation
                        }

                finally:
                    driver.quit()

            except ImportError:
                # Fallback: simulate UI test
                return {
                    "success": True,
                    "action": action,
                    "message": "UI test simulated (selenium not available)",
                    "note": "Install selenium and chromedriver for real UI testing",
                    "explanation": explanation
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"UI test failed: {str(e)}",
                "action": action,
                "explanation": explanation
            }


class LiteLLMAgentHarness:
    """Enhanced agent harness with comprehensive tool capabilities using litellm"""

    def __init__(
        self,
        model_name: str = "claude-3-haiku-20240307",
        container=None,
        base_path: str = "/workspace",
        max_tokens: int = 4000,
        temperature: float = 0.1,
        mern_config: dict = None,
    ):
        self.model_name = model_name
        self.container = container
        self.base_path = base_path
        self.max_tokens = max_tokens
        self.temperature = temperature

        self.enhanced_tools = EnhancedTools(container, base_path, mern_config)

        # Define available tools
        self.tools = [tool for tool in self.enhanced_tools.tools_info]

    def _execute_tool_call(self, tool_call) -> dict:
        """Execute a tool call and return the result"""
        function_name = tool_call.function.name
        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON in tool arguments: {str(e)}"}

        # Route to appropriate tool method
        if function_name == "codebase_search":
            return self.enhanced_tools.codebase_search(
                query=arguments["query"],
                explanation=arguments.get("explanation", ""),
                target_directories=arguments.get("target_directories"),
            )
        elif function_name == "read_file":
            return self.enhanced_tools.read_file(
                target_file=arguments["target_file"],
                explanation=arguments.get("explanation", ""),
                should_read_entire_file=arguments.get("should_read_entire_file", True),
                start_line_one_indexed=arguments.get("start_line_one_indexed", 1),
                end_line_one_indexed_inclusive=arguments.get(
                    "end_line_one_indexed_inclusive", -1
                ),
            )
        elif function_name == "run_terminal_cmd":
            return self.enhanced_tools.run_terminal_cmd(
                command=arguments["command"],
                explanation=arguments.get("explanation", ""),
                is_background=arguments.get("is_background", False),
            )
        elif function_name == "list_dir":
            return self.enhanced_tools.list_dir(
                relative_workspace_path=arguments["relative_workspace_path"],
                explanation=arguments.get("explanation", ""),
            )
        elif function_name == "grep_search":
            return self.enhanced_tools.grep_search(
                query=arguments["query"],
                explanation=arguments.get("explanation", ""),
                case_sensitive=arguments.get("case_sensitive", False),
                include_pattern=arguments.get("include_pattern"),
                exclude_pattern=arguments.get("exclude_pattern"),
            )
        elif function_name == "edit_file":
            return self.enhanced_tools.edit_file(
                target_file=arguments["target_file"],
                instructions=arguments["instructions"],
                edit_type=arguments.get("edit_type", "line_edits"),
                line_edits=arguments.get("line_edits"),
            )
        elif function_name == "search_replace":
            return self.enhanced_tools.search_replace(
                file_path=arguments["file_path"],
                old_string=arguments["old_string"],
                new_string=arguments["new_string"],
            )
        elif function_name == "file_search":
            return self.enhanced_tools.file_search(
                query=arguments["query"], explanation=arguments.get("explanation", "")
            )
        elif function_name == "delete_file":
            return self.enhanced_tools.delete_file(
                target_file=arguments["target_file"],
                explanation=arguments.get("explanation", ""),
            )
        elif function_name == "web_search":
            return self.enhanced_tools.web_search(
                search_term=arguments["search_term"],
                explanation=arguments.get("explanation", ""),
            )
        elif function_name == "create_diagram":
            return self.enhanced_tools.create_diagram(content=arguments["content"])
        elif function_name == "edit_notebook":
            return self.enhanced_tools.edit_notebook(
                target_notebook=arguments["target_notebook"],
                cell_idx=arguments["cell_idx"],
                is_new_cell=arguments["is_new_cell"],
                cell_language=arguments["cell_language"],
                old_string=arguments["old_string"],
                new_string=arguments["new_string"],
            )
        # Backward compatibility with old tool names
        elif function_name == "write_file":
            return self.enhanced_tools.write_file(
                arguments["file_path"], arguments["content"]
            )
        elif function_name == "list_files":
            directory = arguments.get("directory", ".")
            return self.enhanced_tools.list_files(directory)
        elif function_name == "create_directory":
            return self.enhanced_tools.create_directory(arguments["directory_path"])
        elif function_name == "api_call":
            return self.enhanced_tools.api_call(
                method=arguments["method"],
                url=arguments["url"],
                headers=arguments.get("headers"),
                body=arguments.get("body"),
                explanation=arguments.get("explanation", ""),
            )
        elif function_name == "database_query":
            return self.enhanced_tools.database_query(
                query_type=arguments["query_type"],
                collection=arguments["collection"],
                query=arguments.get("query"),
                data=arguments.get("data"),
                explanation=arguments.get("explanation", ""),
            )
        elif function_name == "websocket_test":
            return self.enhanced_tools.websocket_test(
                event_name=arguments["event_name"],
                event_data=arguments.get("event_data"),
                expected_response=arguments.get("expected_response", ""),
                timeout=arguments.get("timeout", 10),
                explanation=arguments.get("explanation", ""),
            )
        elif function_name == "ui_test":
            return self.enhanced_tools.ui_test(
                action=arguments["action"],
                selector=arguments.get("selector", ""),
                text=arguments.get("text", ""),
                url=arguments.get("url", ""),
                timeout=arguments.get("timeout", 10),
                explanation=arguments.get("explanation", ""),
            )
        else:
            return {"error": f"Unknown function: {function_name}"}

    def execute_task(self, task_prompt: str, max_iterations: int = 10) -> dict:
        """Execute a task using the agent with file editing capabilities"""
        messages = [
            {
                "role": "system",
                "content": """
You are a powerful coding assistant with comprehensive development capabilities. You have access to the following tools:

SEARCH & DISCOVERY:
- codebase_search: Semantic search through codebase to find relevant code snippets
- grep_search: Fast regex-based text search with file filtering
- file_search: Fuzzy search for files by name
- list_dir: List directory contents for exploration

FILE OPERATIONS:
- read_file: Read file contents with optional line range support
- edit_file: Propose structured edits to files
- search_replace: Find and replace text in files
- write_file: Create new files or overwrite existing ones
- delete_file: Remove files from the filesystem

EXECUTION & AUTOMATION:
- run_terminal_cmd: Execute shell commands (with background support)
- create_directory: Create directory structures

SPECIALIZED TOOLS:
- edit_notebook: Edit Jupyter notebook cells
- create_diagram: Generate Mermaid diagrams
- web_search: Search the web for information (when available)

MERN STACK TOOLS:
- api_call: Make HTTP requests to test REST API endpoints
- database_query: Execute MongoDB queries (find, insert, update, delete, aggregate)
- websocket_test: Test Socket.IO real-time functionality
- ui_test: Browser automation for React frontend testing (screenshot, click, type, navigate)

WORKFLOW GUIDELINES:
1. Break down complex tasks into steps
2. Use search tools to understand the codebase first
3. Read relevant files before making changes
4. Use appropriate tools for the task (semantic search vs grep vs file search)
5. Provide clear explanations with each tool use
6. Test changes when possible using terminal commands

FOR MERN STACK APPLICATIONS:
1. Identify if you're working with a MERN (MongoDB, Express, React, Node.js) stack
2. Use api_call to test backend endpoints after making changes
3. Use database_query to verify data persistence in MongoDB
4. Use websocket_test for real-time features (chat, notifications, live updates)
5. Use ui_test to verify frontend functionality and user interactions
6. Look for server/ directory (backend), client/ directory (frontend), and package.json files

Always explain your reasoning and approach clearly.
""",
            },
            {"role": "user", "content": task_prompt},
        ]

        iterations = 0
        conversation_history = []
        made_code_changes = False

        while iterations < max_iterations:
            iterations += 1

            try:
                print(f"HARNESS: Iteration {iterations}, making LLM call with model {self.model_name}")
                print(f"HARNESS: Messages length: {len(messages)}")
                print(f"HARNESS: Tools available: {len(self.tools)}")

                # Context window management for Claude (200k token limit with buffer)
                estimated_tokens = estimate_message_tokens(messages)
                print(f"HARNESS: Estimated tokens: {estimated_tokens}")

                if estimated_tokens > 180000:  # Leave 20k buffer for response
                    print(f"HARNESS: Truncating conversation to fit context window")
                    # Keep system message and last 10 messages only
                    system_msg = messages[0]
                    recent_messages = messages[-10:]
                    messages = [system_msg] + recent_messages
                    print(f"HARNESS: Truncated to {len(messages)} messages")

                # Make the LLM call
                # Add safety settings for Gemini to prevent blocking
                extra_params = {}
                if "gemini" in self.model_name.lower():
                    extra_params["safety_settings"] = [
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                    ]

                response = completion(
                    model=self.model_name,
                    messages=messages,
                    tools=self.tools,
                    tool_choice="auto",
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    **extra_params,
                )
                message = response.choices[0].message

                if "gemini" in self.model_name.lower() and message.content is None:
                    message.content = ""

                messages.append(message.dict())

                print(f"HARNESS: Got response. Content length: {len(message.content or '')}")
                print(f"HARNESS: Tool calls: {len(message.tool_calls) if message.tool_calls else 0}")

                # Prepare tool call details for conversation history
                tool_call_details = []
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        # Parse arguments safely
                        try:
                            arguments = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError:
                            arguments = {"error": "Invalid JSON in arguments"}

                        tool_call_details.append(
                            {
                                "tool_call_id": tool_call.id,
                                "function_name": tool_call.function.name,
                                "arguments": arguments,
                            }
                        )

                conversation_history.append(
                    {
                        "iteration": iterations,
                        "message": message.content,
                        "tool_calls_requested": tool_call_details,
                        "tool_results": [],
                    }
                )

                # Check if there are tool calls to execute
                if message.tool_calls:
                    tool_results = []
                    print(f"HARNESS: Processing {len(message.tool_calls)} tool calls")

                    for i, tool_call in enumerate(message.tool_calls):
                        print(f"HARNESS: Tool call {i}: {tool_call.function.name}")

                        # Enhanced logging for edit operations
                        if tool_call.function.name == "edit_file":
                            args = json.loads(tool_call.function.arguments)
                            print(f"HARNESS: Edit target: {args.get('target_file', 'N/A')}")
                            print(f"HARNESS: Edit instructions: {args.get('instructions', 'N/A')[:100]}...")
                            if 'line_edits' in args:
                                print(f"HARNESS: Line edits count: {len(args['line_edits'])}")
                                for j, edit in enumerate(args['line_edits'][:3]):  # Show first 3 edits
                                    edit_type = edit.get('type', 'unknown')
                                    line_num = edit.get('line_number', 'N/A')
                                    print(f"HARNESS: Edit {j}: {edit_type} at line {line_num}")

                        result = self._execute_tool_call(tool_call)
                        print(f"HARNESS: Tool {i} result success: {result.get('success', 'N/A')}")

                        # Enhanced error reporting
                        if 'error' in result:
                            print(f"HARNESS: Tool {i} error: {result['error'][:200]}...")
                            if result.get('syntax_error'):
                                print(f"HARNESS: SYNTAX ERROR at line {result.get('syntax_line', 'N/A')}: {result.get('syntax_msg', 'N/A')}")
                                if 'attempted_changes' in result:
                                    print(f"HARNESS: Attempted changes: {result['attempted_changes']}")
                        elif result.get('success') and 'changes_made' in result:
                            print(f"HARNESS: Changes applied: {result['changes_made'][:3]}")  # Show first 3 changes
                            if tool_call.function.name == "edit_file":
                                made_code_changes = True

                        tool_results.append(
                            {
                                "tool_call_id": tool_call.id,
                                "function_name": tool_call.function.name,
                                "result": result,
                            }
                        )

                        # Add tool result to messages
                        messages.append(
                            {
                                "role": "tool",
                                "content": json.dumps(result),
                                "tool_call_id": tool_call.id,
                            }
                        )

                    conversation_history[-1]["tool_results"] = tool_results
                else:
                    # No tool calls from the model. Enforce completion guard.
                    if not made_code_changes and iterations < max_iterations:
                        print(f"HARNESS: No tool calls yet and no edits made. Prompting agent to implement a change before completion.")
                        messages.append({
                            "role": "system",
                            "content": (
                                "Stop exploring and implement now: open the confirmed file, apply a minimal edit_file "
                                "to the target function, read it back, and continue with small iterative edits until "
                                "the change aligns with the expected behavior/diff."
                            ),
                        })
                        continue
                    # Either edits were made or we hit max iterations; allow completion
                    print(f"HARNESS: No tool calls, task complete at iteration {iterations}")
                    break

            except Exception as e:
                return {
                    "success": False,
                    "error": f"Error during execution: {str(e)}",
                    "iterations": iterations,
                    "conversation_history": conversation_history,
                }

        return {
            "success": made_code_changes,
            "final_response": messages[-1]["content"] if messages else "No response",
            "iterations": iterations,
            "conversation_history": conversation_history,
            "max_iterations_reached": iterations >= max_iterations,
            "made_code_changes": made_code_changes,
            "error": None if made_code_changes else "Run completed without any successful edits",
        }


def test_agent_harness():
    """Test the LiteLLM Agent Harness with a simple task"""
    print("\n=== Testing LiteLLM Agent Harness ===")

    # Create a test harness (without container for local testing)
    harness = LiteLLMAgentHarness(
        model_name="claude-3-haiku-20240307",
        container=None,
        base_path="./test_workspace",
    )

    # Simple test task
    task_prompt = """
Create a simple Python script called 'hello.py' that:
1. Defines a function called greet(name) that returns a greeting message
2. Has a main section that calls greet("World") and prints the result
3. Make sure the file is properly formatted
    """

    print(f"Task: {task_prompt}")
    print("\nExecuting task...")

    result = harness.execute_task(task_prompt, max_iterations=5)

    print(f"\nTask completed. Success: {result['success']}")
    print(f"Iterations used: {result['iterations']}")

    if result["success"]:
        print(f"Final response: {result['final_response']}")
        print(f"\nConversation history ({len(result['conversation_history'])} steps):")
        for i, step in enumerate(result["conversation_history"], 1):
            print(f"  Step {i}: {step['message'][:100]}...")
            if step["tool_calls"]:
                for tool_call in step["tool_calls"]:
                    print(
                        f"    -> {tool_call['function_name']}: {tool_call['result']['success']}"
                    )
    else:
        print(f"Error: {result['error']}")

    return result
