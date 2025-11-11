def run_command_in_container(container, command: list, stream: bool = False) -> dict:
    """Execute command in container and return results
    Note: exit code is None if stream is true
    """

    try:
        exit_code, output = container.exec_run(cmd=command, stream=stream)

        # Handle streaming output
        if stream:
            output_lines = []
            for chunk in output:
                if chunk:
                    line = chunk.decode("utf-8").strip()
                    output_lines.append(line)
                    # print(line)  # Real-time output

            output = "\n".join(output_lines)
        else:
            output = output.decode("utf-8") if output else ""

        return {
            "success": exit_code == 0,
            "exit_code": exit_code,
            "output": output,
            "command": command,
        }

    except Exception as e:
        return {"success": False, "exit_code": -1, "output": str(e), "command": command}
