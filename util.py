from typing import Any, Dict


def parse_task_description(task_description_text: str) -> Dict[str, Any]:
    """
    Parse a well-structured task description into a dictionary.

    Supports two formats:

    Format 1 (Original):
    task_description |
      Task: task_name
      Task ID: 001

      Instructions: detailed instructions...

    author_name: Author Name
    author_email: email@example.com
    difficulty: easy|medium|hard
    category: category_name
    tags: <tag1> <tag2>
    parser_name: <pytest>

    Format 2 (YAML-style):
    task_description: |
      Task: Task Name
      Task ID: task-id
      Objective:
        Description here...
    """
    result = {}

    lines = task_description_text.strip().split("\n")

    is_yaml_format = False
    if lines and lines[0].strip().startswith("task_description:"):
        is_yaml_format = True

    if is_yaml_format:
        in_task_description = False
        task_content_lines = []
        remaining_lines = []

        for i, line in enumerate(lines):
            stripped = line.strip()

            if stripped.startswith("task_description:"):
                in_task_description = True
                continue
            elif in_task_description:
                if line.startswith("  ") or not stripped:
                    task_content_lines.append(line[2:] if line.startswith("  ") else "")
                else:
                    # Save remaining lines for parsing
                    remaining_lines = lines[i:]
                    break

        # Parse task_description content
        task_content = "\n".join(task_content_lines).strip()
        task_lines = task_content.split("\n")

        objective_lines = []
        in_objective = False

        for task_line in task_lines:
            task_line = task_line.strip()
            if not task_line:
                continue

            if ":" in task_line and not in_objective:
                key, value = task_line.split(":", 1)
                key = key.strip().lower().replace(" ", "_")
                value = value.strip()

                if key == "objective" or key == "instructions":
                    in_objective = True
                    if value:
                        objective_lines.append(value)
                else:
                    result[key] = value
            elif in_objective:
                objective_lines.append(task_line)

        if objective_lines:
            result["instructions"] = "\n".join(objective_lines).strip()

        # Parse remaining key-value pairs outside task_description block
        for line in remaining_lines:
            line = line.strip()
            if not line:
                continue

            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower().replace(" ", "_")
                value = value.strip()

                # Handle special parser_name format
                if key == "parser_name" and value.startswith("<"):
                    result[key] = value.strip("<>")
                else:
                    result[key] = value

        result.setdefault("author_name", "System")
        result.setdefault("author_email", "system@example.com")
        result.setdefault("difficulty", "medium")
        result.setdefault("category", "Backend")
        result.setdefault("tags", ["mern"])
        result.setdefault("parser_name", "jest")  # Default for MERN projects

        return result

    # Parse key-value pairs (both formats)
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Handle different formats
        if ":" in line:
            # Split on first colon to handle values with colons
            key, value = line.split(":", 1)
            key = key.strip().lower().replace(" ", "_")
            value = value.strip()

            # Handle special cases
            if key == "tags" and value.startswith("<"):
                # Parse tags like "<tag1> <tag2>"
                result[key] = [tag.strip("<>") for tag in value.split()]
            elif key == "parser_name" and value.startswith("<"):
                # Parse parser name like "<pytest>"
                result[key] = value.strip("<>")
            else:
                result[key] = value

    # Handle multi-line instructions
    instructions_lines = []
    in_instructions = False

    for line in lines:
        line = line.strip()
        if line.startswith("Instructions:"):
            in_instructions = True
            # Extract the part after "Instructions:"
            instructions_lines.append(line.split(":", 1)[1].strip())
        elif (
            in_instructions
            and line
            and not line.startswith(
                ("author_", "difficulty:", "category:", "tags:", "parser_name:")
            )
        ):
            instructions_lines.append(line)

    if instructions_lines:
        result["instructions"] = "\n".join(instructions_lines).strip()

    return result


# Test function for the parser
def test_task_description_parser():
    """Test the task description parser with sample data"""

    # Sample task description content
    sample_content = """task_description |
  Task: add is_odd
  Task ID: 001

  Instructions: add is_odd and have main.py print out whether or not the random
  number is odd.

author_name: Andrew Yu
author_email: <you@example.com>
difficulty: easy
category: Feature
tags: <python>
parser_name: <pytest>"""

    parsed = parse_task_description(sample_content)

    print("Parsed task description:")
    print(f"Task: {parsed.get('task')}")
    print(f"Task ID: {parsed.get('task_id')}")
    print(f"Instructions: {parsed.get('instructions')}")
    print(f"Author: {parsed.get('author_name')}")
    print(f"Difficulty: {parsed.get('difficulty')}")
    print(f"Tags: {parsed.get('tags')}")
    print(f"Parser: {parsed.get('parser_name')}")

    return parsed


if __name__ == "__main__":
    test_task_description_parser()
