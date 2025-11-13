# IDE Arena

IDE Arena is a comprehensive framework for evaluating AI IDE agents on real-world software engineering tasks across diverse technology stacks. We define IDE agents as AI models operating in a chat-based IDE environment with access to the same tools available in agent-enabled IDEs like Cursor. While adoption of agent-enabled IDEs is rapidly growing, there is no existing benchmark to rigorously test how well models perform as IDE agents in practice.


## Quick Start

### Prerequisites

- Python with `uv` package manager
- Docker running

### Running Benchmarks

**Oracle Agent (Golden Solution)**

```bash
uv run main.py bench --dataset /path_to_directory --agent oracle --model oracle --task-id name_of_task
```

**AI Agent (Real Model)**

```bash
uv run main.py bench --dataset /path_to_directory --agent harness --model litellm_model_name --task-id name_of_task
```

## Environment Setup

Set your API keys:

```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
...
```

You can now run with any LiteLLM supported model tag via litellm_model_name

## Web Interface

Start the Next.js dashboard to view traces and results:

```bash
npm run dev
```

## Dataset Structure

Each dataset must contain the following required files and directories:

```
dataset/
├── Dockerfile                         # Container definition for the task environment
├── docker-compose.yaml                # Docker compose configuration (or compose.yaml, docker-compose.yml)
├── run_tests.sh                       # Test execution script
└── tasks/                             # Task definitions directory
    ├── task-name-1/
    │   ├── task_description.txt        # Task description and instructions
    │   ├── task_diff.txt               # Golden solution diff (for oracle mode)
    │   ├── task_tests.py               # Task-specific test file
    │   ├── run-tests.sh                # Task-specific test runner script
    │   └── docker-compose.yaml         # Task-specific container configuration
    ├── task-name-2/
    │   ├── task_description.txt
    │   ├── task_diff.txt
    │   ├── task_tests.py
    │   ├── run-tests.sh
    │   └── docker-compose.yaml
    └── ...
```
