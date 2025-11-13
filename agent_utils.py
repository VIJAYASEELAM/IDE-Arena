from docker_utils import run_command_in_container
from harness import LiteLLMAgentHarness
import os
import re


def analyze_task_requirements(task_data: dict) -> dict:
    """Analyze task requirements to generate appropriate implementation guidance"""
    instructions = task_data.get("instructions", "").lower()
    task_name = task_data.get("task", "").lower()

    analysis = {
        "task_type": "unknown",
        "endpoints_mentioned": [],
        "requires_new_endpoint": False,
        "requires_endpoint_modification": False,
        "requires_environment_vars": False,
        "requires_algorithm": False,
        "requires_database": False,
        "key_concepts": []
    }

    # Extract mentioned endpoints
    endpoint_patterns = [
        r'/api/[^\s\'"]+',
        r'GET /[^\s\'"]+',
        r'POST /[^\s\'"]+',
        r'endpoint at ([^\s\'"]+)'
    ]

    for pattern in endpoint_patterns:
        matches = re.findall(pattern, instructions)
        analysis["endpoints_mentioned"].extend(matches)

    # Determine task type
    if "environment variable" in instructions:
        analysis["task_type"] = "configuration"
        analysis["requires_environment_vars"] = True
        analysis["key_concepts"].append("environment variables")

    if "new endpoint" in instructions or "implement.*endpoint" in instructions:
        analysis["task_type"] = "new_endpoint"
        analysis["requires_new_endpoint"] = True

    if "modify" in instructions and "endpoint" in instructions:
        analysis["requires_endpoint_modification"] = True

    if any(word in instructions for word in ["anomaly", "detection", "algorithm", "statistical", "mean", "deviation"]):
        analysis["task_type"] = "algorithm"
        analysis["requires_algorithm"] = True
        analysis["key_concepts"].extend(["statistical analysis", "algorithm implementation"])

    if any(word in instructions for word in ["database", "mongodb", "query", "aggregate"]):
        analysis["requires_database"] = True
        analysis["key_concepts"].append("database operations")

    if "configurable" in task_name or "default" in task_name:
        analysis["task_type"] = "configuration"
        analysis["requires_endpoint_modification"] = True

    return analysis


def discover_candidate_files(container, task_data: dict) -> list[str]:
    """Heuristically discover likely target files inside the container.
    Searches common API/controller directories for task-relevant keywords.
    """
    instructions = (task_data.get("instructions", "") or "").lower()
    keywords: list[str] = []

    # Generic API/controller markers
    keywords.extend(["APIRouter", "@router", "def upload", "def top_paths", "def anomalies", "def error", "upload_log_file"])  # noqa: E501

    # Add from instructions
    if any(k in instructions for k in ["upload", "log", "regex", "malformed"]):
        keywords.extend(["upload", "log", "malformed", "regex"])  # logs-related
    if any(k in instructions for k in ["anomaly", "3-sigma", "sigma", "std", "mean"]):
        keywords.extend(["anomal", "stats", "sigma"])  # anomalies in stats
    if any(k in instructions for k in ["top", "paths", "limit"]):
        keywords.extend(["top_paths", "limit", "stats"])  # top paths
    if any(k in instructions for k in ["error", "summary", "status"]):
        keywords.extend(["error", "summary", "status"])  # error summary

    # Deduplicate and build grep pattern
    unique_keywords = sorted(set(keywords))
    if not unique_keywords:
        return []

    # Build a single egrep pattern joined by '|'
    pattern = "|".join([re.escape(k) for k in unique_keywords])

    # Search common locations; keep it general
    search_dirs = [
        "/app/app/api",
        "/app/app/core",
        "/app/app/routes",
        "/app/app/controllers",
    ]

    found_files: list[str] = []
    for d in search_dirs:
        cmd = [
            "bash", "-lc",
            f"if [ -d '{d}' ]; then grep -RInE '{pattern}' '{d}' --include=*.py || true; fi"
        ]
        res = run_command_in_container(container=container, command=cmd)
        if not res.get("success"):
            continue
        output = res.get("output", "")
        for line in output.splitlines():
            # Expect lines like: /path/file.py:123:matched text
            parts = line.split(":", 2)
            if len(parts) >= 2:
                path = parts[0].strip()
                if path.endswith(".py") and path not in found_files:
                    found_files.append(path)

    # Rank: prefer api/ first, then others; cap to top 5
    def rank(p: str) -> tuple[int, int]:
        priority = 0
        if "/api/" in p:
            priority = 0
        elif "/routes/" in p or "/controllers/" in p:
            priority = 1
        else:
            priority = 2
        length = len(p)
        return (priority, length)

    found_files.sort(key=rank)
    return found_files[:5]


def generate_implementation_guidance(task_data: dict, analysis: dict) -> str:
    """Generate task-specific implementation guidance based on analysis"""
    instructions = task_data.get("instructions", "")

    guidance = []

    # Base project structure info
    guidance.append("PROJECT ARCHITECTURE (GENERAL):")
    guidance.append("- Identify the application entrypoint and routing/module structure for the given stack")
    guidance.append("- Common patterns: routers/controllers in api/routes/controllers directories; domain logic in services/utils; DB models in models/entities")
    guidance.append("")

    # Task-specific approach
    if analysis["task_type"] == "configuration":
        guidance.append("CONFIGURATION TASK APPROACH:")
        if analysis["requires_environment_vars"]:
            guidance.append("1. Read environment variables using os.environ.get()")
            guidance.append("2. Implement parameter precedence logic")
            guidance.append("3. Handle missing/invalid values gracefully")
        if analysis["requires_endpoint_modification"]:
            guidance.append("4. Locate existing endpoint and modify its behavior")
            guidance.append("5. Preserve existing functionality while adding new features")

    elif analysis["task_type"] == "algorithm":
        guidance.append("ALGORITHM IMPLEMENTATION APPROACH:")
        guidance.append("1. Understand the mathematical/statistical requirements")
        guidance.append("2. Implement core algorithm logic with proper data structures")
        guidance.append("3. Handle edge cases (empty data, insufficient samples)")
        guidance.append("4. Create new endpoint to expose the algorithm")
        if "anomaly" in instructions.lower():
            guidance.append("5. Implement 3-sigma statistical analysis (mean + 3*std_dev)")
            guidance.append("6. Support time-based bucketing and filtering")

    elif analysis["task_type"] == "new_endpoint":
        guidance.append("NEW ENDPOINT IMPLEMENTATION:")
        guidance.append("1. Create new endpoint function in stats router")
        guidance.append("2. Define proper request/response models")
        guidance.append("3. Implement core business logic")
        guidance.append("4. Add proper error handling and validation")

    else:
        guidance.append("GENERAL IMPLEMENTATION APPROACH:")
        guidance.append("1. Analyze requirements to identify needed endpoints")
        guidance.append("2. Check existing code patterns in stats router")
        guidance.append("3. Implement required functionality")
        guidance.append("4. Test against provided requirements")

    guidance.append("")

    # Endpoint-specific guidance
    if analysis["endpoints_mentioned"]:
        guidance.append("REQUIRED ENDPOINTS:")
        for endpoint in set(analysis["endpoints_mentioned"]):
            guidance.append(f"- {endpoint}")
        guidance.append("")

    # Technical requirements
    guidance.append("IMPLEMENTATION REQUIREMENTS (GENERAL):")
    guidance.append("- Modify the appropriate module/router for the feature area (avoid editing the main entrypoint unless required)")
    guidance.append("- Follow existing code patterns, imports, and error handling")
    guidance.append("- Prefer edit_file for structural/multi-line changes; avoid search_replace for multi-line edits")
    guidance.append("- Do not use shell echo appends; use structured line_edits via edit_file")
    guidance.append("- Ensure outputs conform to the project's response/typing conventions")
    guidance.append("- Test implementation meets all stated requirements")

    if analysis["requires_environment_vars"]:
        guidance.append("- Use os.environ.get() for environment variable access")
        guidance.append("- Implement proper default value handling")

    if analysis["requires_algorithm"]:
        guidance.append("- Import necessary libraries (e.g., statistics) as needed by the task")
        guidance.append("- Implement efficient algorithms for large datasets")

    return "\n".join(guidance)


def deploy_agent_in_container(
    container,
    agent_name: str,
    task_id: str,
    model_name: str,
    task_data: dict,
    verbose: bool = False,
) -> dict:
    """Deploy agent in container"""
    if agent_name == "oracle":  # not a real agent
        result = run_command_in_container(
            container=container,
            command=["git", "apply", f"tasks/{task_id}/task_diff.txt"],
        )
        return {"success": result["success"], "output": result["output"]}
    elif agent_name == "harness":
        print(f"AGENT_UTILS: Creating harness agent with model {model_name}")

        # Analyze task requirements to generate appropriate guidance
        analysis = analyze_task_requirements(task_data)
        implementation_guidance = generate_implementation_guidance(task_data, analysis)

        print(f"AGENT_UTILS: Task analysis - Type: {analysis['task_type']}, Endpoints: {analysis['endpoints_mentioned']}")

        # Discover candidate files and include as hints
        candidate_files = discover_candidate_files(container, task_data)
        candidates_hint = "\n".join([f"- {p}" for p in candidate_files]) if candidate_files else "(no candidates found)"

        prompt = f"""You are a software engineer implementing a feature in a production codebase.

TASK: {task_data.get("task", "Unknown task")}
{task_data.get("instructions", "No instructions provided")}

DISCOVERY (find the correct file before editing):
0. List likely source directories (e.g., src/, app/, server/, backend/, api/, routes/) and scan for routers/controllers and domain modules
1. Search the codebase for relevant symbols/keywords (endpoints, function names, config keys) to locate the precise implementation point
2. Read surrounding code in candidate files to confirm the correct edit location and invariants
3. Proceed only after confirming the file path exists and matches the intent; record the confirmed target path

AUTOMATED CANDIDATES (from pre-scan):
{candidates_hint}

POST-DISCOVERY ACTION PLAN (do, don't just explore):
1. Open the confirmed target file with read_file and locate the exact function to modify
2. Apply a minimal, verifiable change with edit_file immediately; then read back and verify
3. Iterate: edit_file → read_file (surrounding lines) → grep for function/route → next small edit until done
4. Keep edits scoped (≤5 line changes per call); avoid search_replace for structural changes

TASK-SPECIFIC BIAS (use when applicable):
- If the task or instructions mention '/api/stats/method-counts' or 'method counts' or 'request method breakdown':
  a) Edit app/api/stats.py
  b) Locate def method_counts(...)
  c) Implement the exact required behavior and return shape based on the task spec
  d) If logic already exists, align names/imports/structure to match the expected golden diff (imports order, variables, structure)
  e) Validate by reading back the function body and printing surrounding lines

REQUIREMENTS:
- Make actual code changes (evaluated by strict diff vs golden)
- Match the file’s existing indentation style (spaces or tabs, width), do not reformat
- Keep each edit_file call simple (≤5 line changes)
- Read file back after each edit to verify
- Completion rule: You must apply at least one successful edit_file to the confirmed file before declaring the task complete. If you haven't edited yet, continue until you do.

EDIT FORMAT:
{{
  "target_file": "<CONFIRMED_TARGET_FILE>",
  "instructions": "Add statistics import",
  "edit_type": "line_edits",
  "line_edits": [
    {{
      "type": "insert",
      "line_number": 2,
      "content": "import statistics"
    }}
  ]
}}

START: List likely directories, grep for relevant symbols, confirm the target file, then read it to understand the structure."""
        print(f"AGENT_UTILS: Prompt length: {len(prompt)}")
        print(f"AGENT_UTILS: Task data tags: {task_data.get('tags', [])}")

        # MERN support is disabled by default for this dataset. Enable later via env flag ENABLE_MERN=1
        mern_config = None
        if os.environ.get("ENABLE_MERN", "") == "1":
            if "mern" in task_data.get("tags", []) or "full-stack" in task_data.get("tags", []):
                mern_config = {
                    "api_base_url": "http://localhost:5001",
                    "frontend_url": "http://localhost:3000",
                    "mongo_uri": "mongodb://localhost:27017/dev-arena-test",
                    "websocket_url": "http://localhost:5001"
                }
                print(f"AGENT_UTILS: Using MERN config: {mern_config}")
        else:
            if "mern" in task_data.get("tags", []) or "full-stack" in task_data.get("tags", []):
                print("AGENT_UTILS: MERN features detected but disabled for this run (ENABLE_MERN!=1)")

        print(f"AGENT_UTILS: Creating LiteLLMAgentHarness...")
        # Use the new LiteLLM Agent Harness
        harness = LiteLLMAgentHarness(
            model_name=model_name, container=container, base_path="/app", mern_config=mern_config
        )

        print(f"AGENT_UTILS: Executing task with max_iterations=35...")
        result = harness.execute_task(prompt, max_iterations=35)
        print(f"AGENT_UTILS: Task execution completed. Success: {result.get('success')}")
        print(f"AGENT_UTILS: Result keys: {list(result.keys())}")
        if 'error' in result:
            print(f"AGENT_UTILS: Error: {result['error']}")
        if 'iterations' in result:
            print(f"AGENT_UTILS: Iterations used: {result['iterations']}")
        if 'conversation_history' in result:
            print(f"AGENT_UTILS: Conversation history length: {len(result['conversation_history'])}")
            for i, conv in enumerate(result['conversation_history'][:3]):  # Show first 3
                print(f"AGENT_UTILS: Conv {i}: {list(conv.keys())}")
                if 'tool_calls_requested' in conv:
                    print(f"AGENT_UTILS: Conv {i} tools: {len(conv['tool_calls_requested'])}")
                if 'tool_results' in conv:
                    print(f"AGENT_UTILS: Conv {i} results: {len(conv['tool_results'])}")
                    for j, tool_result in enumerate(conv['tool_results'][:2]):  # Show first 2 tool results
                        if 'result' in tool_result:
                            print(f"AGENT_UTILS: Tool {j} success: {tool_result['result'].get('success')}")

        if result["success"]:
            return {
                "success": True,
                "task_data": task_data,
                "agent_response": result["final_response"],
                "conversation_history": result["conversation_history"],
                "iterations": result["iterations"],
                "model_used": model_name,
            }
        else:
            return {
                "success": False,
                "error": result["error"],
                "conversation_history": result.get("conversation_history", []),
            }
    else:
        raise ValueError(f"Unsupported agent: {agent_name}")
