from enum import Enum

class Model(Enum):
    GPT_4o = "openai/gpt-4o"
    GPT_5 = "openai/gpt-5"
    GPT_4o_MINI = "openai/gpt-4o-mini"
    GPT_4o_2024_08_06 = "openai/gpt-4o-2024-08-06"
    GPT_4o_2024_05_13 = "openai/gpt-4o-2024-05-13"
    CLAUDE_3_5_SONNET = "anthropic/claude-3-5-sonnet-20241022"
    CLAUDE_3_5_HAIKU = "anthropic/claude-3-5-haiku-20241022"
    CLAUDE_3_OPUS = "anthropic/claude-3-opus-20240229"
    CLAUDE_3_7_SONNET = "anthropic/claude-3-7-sonnet-20250219"
    GEMINI_1_5_FLASH = "gemini/gemini-1.5-flash"


class TestStatus(Enum):
    FAILED = "FAILED"
    PASSED = "PASSED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"
    XFAIL = "XFAIL"
