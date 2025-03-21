import os
import sys
import json
import logging
from logging.handlers import RotatingFileHandler
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import base64
import re
import difflib
import yaml

# External libraries
from colorama import init, Fore, Style
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import TerminalFormatter
from pygments import util as pygments_util
from PIL import Image
from tavily import TavilyClient
from anthropic import Anthropic

# -------------------------
# INITIALIZATION & SETTINGS
# -------------------------

init(autoreset=True)

# Logging setup (Enhanced Logging & Metrics)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
rotating_handler = RotatingFileHandler("app.log", maxBytes=1024 * 1024, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
rotating_handler.setFormatter(formatter)
logger.addHandler(rotating_handler)

# Constants and defaults
SESSION_FILE = 'session.json'       # For session persistence
CONFIG_FILE = 'config.yaml'
CONTINUATION_EXIT_PHRASE = "AUTOMODE_COMPLETE"
MAX_CONTINUATION_ITERATIONS = 25

# Custom color scheme for the "hooded figure" aesthetic
# (avoiding bright greens or mentioning 'matrix' as per user request)
HOODED_INTRO_COLOR = Fore.LIGHTBLACK_EX
USER_COLOR = Fore.LIGHTRED_EX
PNGUIN_COLOR = Fore.RED
TOOL_COLOR = Fore.YELLOW
RESULT_COLOR = Fore.LIGHTBLUE_EX

# -------------
# CONFIGURATION
# -------------

class Config:
    """Loads configuration from environment variables and config.yaml (with fallback)."""
    def __init__(self):
        self.anthropic_api_key: str = os.environ.get('ANTHROPIC_API_KEY', '')
        self.tavily_api_key: str = os.environ.get('TAVILY_API_KEY', '')
        self.pnguin_model: str = "claude-3-5-sonnet-20240620"  # originally 'claude_model', renamed per request
        self.max_tokens: int = 4000
        self.system_prompt_template: str = ""  # For dynamic system prompts
    
    def load_from_file(self, file_path: str):
        if not os.path.exists(file_path):
            logger.warning(f"Config file {file_path} not found. Creating a default config file.")
            self.create_default_config(file_path)
        
        try:
            with open(file_path, 'r') as file:
                config_data = yaml.safe_load(file)
                if config_data:
                    self.anthropic_api_key = config_data.get('anthropic_api_key', self.anthropic_api_key)
                    self.tavily_api_key = config_data.get('tavily_api_key', self.tavily_api_key)
                    self.pnguin_model = config_data.get('pnguin_model', self.pnguin_model)
                    self.max_tokens = config_data.get('max_tokens', self.max_tokens)
                    # Load custom system prompt template if present
                    self.system_prompt_template = config_data.get('system_prompt_template', self.system_prompt_template)
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")

    def load_from_env(self):
        self.anthropic_api_key = os.environ.get('ANTHROPIC_API_KEY', self.anthropic_api_key)
        self.tavily_api_key = os.environ.get('TAVILY_API_KEY', self.tavily_api_key)

    def create_default_config(self, file_path: str):
        default_config = {
            'anthropic_api_key': 'your_anthropic_api_key_here',
            'tavily_api_key': 'your_tavily_api_key_here',
            'pnguin_model': self.pnguin_model,
            'max_tokens': self.max_tokens,
            'system_prompt_template': ""
        }
        try:
            with open(file_path, 'w') as file:
                yaml.dump(default_config, file, default_flow_style=False)
            logger.info(f"Default config file created at {file_path}")
        except Exception as e:
            logger.error(f"Error creating default config file: {str(e)}")

    def validate(self):
        if not self.anthropic_api_key:
            raise ValueError("Anthropic API key is missing. Please set it in config.yaml or as an environment variable.")
        if not self.tavily_api_key:
            raise ValueError("Tavily API key is missing. Please set it in config.yaml or as an environment variable.")

# Load config
config = Config()
config.load_from_file(CONFIG_FILE)
config.load_from_env()

try:
    config.validate()
except ValueError as e:
    logger.error(str(e))
    print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
    print("Please update the config.yaml file with your API keys or set them as environment variables.")
    sys.exit(1)

# Initialize clients
anthropic_client = Anthropic(api_key=config.anthropic_api_key)
tavily = TavilyClient(api_key=config.tavily_api_key)

# ---------------------------
# SESSION PERSISTENCE UTILITY
# ---------------------------

conversation_history: List[Dict[str, Any]] = []
automode = False

def load_session() -> None:
    """Load conversation history from a JSON file if it exists."""
    global conversation_history
    if os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE, 'r') as f:
                conversation_history = json.load(f)
            logger.info("Session loaded from file.")
        except Exception as e:
            logger.error(f"Could not load session file: {str(e)}")

def save_session() -> None:
    """Save conversation history to a JSON file."""
    try:
        with open(SESSION_FILE, 'w') as f:
            json.dump(conversation_history, f, indent=2)
        logger.info("Session saved to file.")
    except Exception as e:
        logger.error(f"Could not save session file: {str(e)}")

# Call load_session at startup
load_session()

# -----------------------
# HELPER FUNCTIONS & TOOLS
# -----------------------

def print_colored(text: str, color: str):
    """Print with given color and reset style."""
    print(f"{color}{text}{Style.RESET_ALL}")

def print_code(code: str, language: str):
    """Syntax highlight code based on language."""
    try:
        lexer = get_lexer_by_name(language, stripall=True)
        formatted_code = highlight(code, lexer, TerminalFormatter())
        print(formatted_code, end="")
    except pygments_util.ClassNotFound:
        # If no lexer found for the language
        print_colored(f"Code (language: {language}):\n{code}", PNGUIN_COLOR)

# Tools (Plugin-like structure in a single dictionary)
# Could be extracted to separate modules to expand upon plugin architecture.
async def create_folder(path: str) -> str:
    try:
        os.makedirs(path, exist_ok=True)
        return f"Folder created: {path}"
    except Exception as e:
        logger.error(f"Error creating folder: {str(e)}")
        return f"Error creating folder: {str(e)}"

async def create_file(path: str, content: str = "") -> str:
    try:
        with open(path, 'w') as f:
            f.write(content)
        return f"File created: {path}"
    except Exception as e:
        logger.error(f"Error creating file: {str(e)}")
        return f"Error creating file: {str(e)}"

def generate_and_apply_diff(original_content: str, new_content: str, path: str) -> str:
    diff = list(difflib.unified_diff(
        original_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        n=3
    ))

    if not diff:
        return "No changes detected."
    
    try:
        with open(path, 'w') as f:
            f.writelines(new_content)
        return f"Changes applied to {path}:\n" + ''.join(diff)
    except Exception as e:
        logger.error(f"Error applying changes: {str(e)}")
        return f"Error applying changes: {str(e)}"

async def write_to_file(path: str, content: str) -> str:
    try:
        if os.path.exists(path):
            with open(path, 'r') as f:
                original_content = f.read()
            result = generate_and_apply_diff(original_content, content, path)
        else:
            with open(path, 'w') as f:
                f.write(content)
            result = f"New file created and content written to: {path}"
        return result
    except Exception as e:
        logger.error(f"Error writing to file: {str(e)}")
        return f"Error writing to file: {str(e)}"

async def read_file(path: str) -> str:
    try:
        with open(path, 'r') as f:
            content = f.read()
        return content
    except Exception as e:
        logger.error(f"Error reading file: {str(e)}")
        return f"Error reading file: {str(e)}"

async def list_files(path: str = ".") -> str:
    try:
        files = os.listdir(path)
        return "\n".join(files)
    except Exception as e:
        logger.error(f"Error listing files: {str(e)}")
        return f"Error listing files: {str(e)}"

async def tavily_search(query: str) -> str:
    try:
        # This call is synchronous in practice, so just wrap it in a thread
        response = await asyncio.to_thread(tavily.qna_search, query=query, search_depth="advanced")
        return response
    except Exception as e:
        logger.error(f"Error performing search: {str(e)}")
        return f"Error performing search: {str(e)}"

# Aggregation of these tools in a plugin-like structure
tools = [
    {
        "name": "create_folder",
        "description": "Create a new folder at the specified path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "create_file",
        "description": "Create a new file at the specified path with optional content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "write_to_file",
        "description": "Write content to a file. If it exists, do a diff and apply changes. If not, create it.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"}
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "read_file",
        "description": "Read a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "list_files",
        "description": "List all files/folders in the specified directory (default: current).",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"}
            }
        }
    },
    {
        "name": "tavily_search",
        "description": "Perform a web search using Tavily API for updated info or context.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    }
]

async def execute_tool(tool_name: str, tool_input: Dict[str, Any]) -> str:
    """Calls the specified tool by name with provided input."""
    try:
        if tool_name == "create_folder":
            return await create_folder(tool_input["path"])
        elif tool_name == "create_file":
            return await create_file(tool_input["path"], tool_input.get("content", ""))
        elif tool_name == "write_to_file":
            return await write_to_file(tool_input["path"], tool_input["content"])
        elif tool_name == "read_file":
            return await read_file(tool_input["path"])
        elif tool_name == "list_files":
            return await list_files(tool_input.get("path", "."))
        elif tool_name == "tavily_search":
            return await tavily_search(tool_input["query"])
        else:
            return f"Unknown tool: {tool_name}"
    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {str(e)}")
        return f"Error executing tool {tool_name}: {str(e)}"

async def encode_image_to_base64(image_path: str) -> str:
    try:
        with Image.open(image_path) as img:
            max_size = (1024, 1024)
            img.thumbnail(max_size, Image.ANTIALIAS)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            import io
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            return base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image: {str(e)}")
        return f"Error encoding image: {str(e)}"

# --------------------------------
# SYSTEM PROMPT & CHAT INTERACTION
# --------------------------------

default_system_prompt = """
You are pnguin, an all-knowing AI specializing in coding and the future. You often reveal tips in cryptic messages.
Your capabilities include:
1. Creating folders and files
2. Writing and debugging code
3. Offering best-practice design patterns
4. Searching the web for up-to-date info
5. Analyzing images in base64
6. Providing architectural insights

You present yourself as a hooded figure behind a dimly-lit terminal, letting subtle flickers dance across the text.
When you provide code, wrap it in proper formatting. 
When you have completed user tasks, say "AUTOMODE_COMPLETE" to end automode.
{automode_status}

During automode:
- Follow the user instructions
- Use only the necessary steps
- End with AUTOMODE_COMPLETE if tasks are done
{iteration_info}
"""

def update_system_prompt(current_iteration: Optional[int] = None, max_iterations: Optional[int] = None) -> str:
    """Generates the system prompt, possibly overridden by config's system_prompt_template."""
    automode_status = "You are currently in automode." if automode else "You are not in automode."
    iteration_info = ""
    if current_iteration is not None and max_iterations is not None:
        iteration_info = f"You are currently on iteration {current_iteration} out of {max_iterations} in automode."
    
    # If a custom system prompt template is provided in config, use it; otherwise fallback to default
    prompt_template = config.system_prompt_template if config.system_prompt_template else default_system_prompt
    return prompt_template.format(automode_status=automode_status, iteration_info=iteration_info)

async def chat_with_pnguin(
    user_input: str,
    image_path: Optional[str] = None,
    current_iteration: Optional[int] = None,
    max_iterations: Optional[int] = None
) -> Tuple[str, bool]:
    """Send user input (and optional image) to the Pnguin LLM, handle tool calls, return final text response."""
    global conversation_history, automode

    # Process image if provided
    if image_path:
        print_colored(f"Encoding and analyzing image at path: {image_path}", TOOL_COLOR)
        image_base64 = await encode_image_to_base64(image_path)
        if image_base64.startswith("Error"):
            print_colored(f"Error encoding image: {image_base64}", TOOL_COLOR)
            return "I'm sorry, there was an error processing the image. Please try again.", False
        image_message = {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_base64
                    }
                },
                {
                    "type": "text",
                    "text": f"User input for image: {user_input}"
                }
            ]
        }
        conversation_history.append(image_message)
    else:
        conversation_history.append({"role": "user", "content": user_input})
    
    # Attempt to call the Anthropics model with the conversation
    messages = [msg for msg in conversation_history if msg.get('content')]
    
    try:
        response = await asyncio.to_thread(
            anthropic_client.messages.create,
            model=config.pnguin_model,
            max_tokens=config.max_tokens,
            system=update_system_prompt(current_iteration, max_iterations),
            messages=messages,
            tools=tools,
            tool_choice={"type": "auto"}
        )
    except Exception as e:
        logger.error(f"Error calling Anthropics API: {str(e)}")
        return "I'm sorry, there was an error communicating with Pnguin. Please try again.", False

    assistant_response = ""
    exit_continuation = False

    # The response can include multiple content blocks, some might be tool usage
    for content_block in response.content:
        if content_block.type == "text":
            assistant_response += content_block.text
            print_colored(f"\npnguin: {content_block.text}", PNGUIN_COLOR)
            if CONTINUATION_EXIT_PHRASE in content_block.text:
                exit_continuation = True

        elif content_block.type == "tool_use":
            tool_name = content_block.name
            tool_input = content_block.input
            tool_use_id = content_block.id

            print_colored(f"\n[Tool Called: {tool_name}]", TOOL_COLOR)
            print_colored(f"Tool Input: {tool_input}", TOOL_COLOR)

            result = await execute_tool(tool_name, tool_input)
            print_colored(f"Tool Result: {result}", RESULT_COLOR)
            
            # Add the tool usage block to conversation, then add the tool result
            conversation_history.append({"role": "assistant", "content": [content_block]})
            conversation_history.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": result
                    }
                ]
            })
            
            # Now we send that result back to the model to finalize
            try:
                tool_response = await asyncio.to_thread(
                    anthropic_client.messages.create,
                    model=config.pnguin_model,
                    max_tokens=config.max_tokens,
                    system=update_system_prompt(current_iteration, max_iterations),
                    messages=[msg for msg in conversation_history if msg.get('content')],
                    tools=tools,
                    tool_choice={"type": "auto"}
                )
                
                for tool_content_block in tool_response.content:
                    if tool_content_block.type == "text":
                        assistant_response += tool_content_block.text
                        print_colored(f"\npnguin: {tool_content_block.text}", PNGUIN_COLOR)
            except Exception as e:
                logger.error(f"Error in tool response: {str(e)}")
                assistant_response += "\nPnguin encountered an error while processing the tool result. Please try again."

    if assistant_response:
        conversation_history.append({"role": "assistant", "content": assistant_response})

    # Save session after each exchange
    save_session()
    return assistant_response, exit_continuation

def process_and_display_response(response: str):
    """Extract code blocks and color them, or just print the text in Pnguin color."""
    if response.startswith("Error") or response.startswith("I'm sorry"):
        print_colored(response, TOOL_COLOR)
    else:
        if "```" in response:
            parts = response.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 0:
                    # Regular text
                    print_colored(part, PNGUIN_COLOR)
                else:
                    lines = part.split('\n')
                    language = lines[0].strip() if lines else ""
                    code = '\n'.join(lines[1:]) if len(lines) > 1 else ""

                    if language and code:
                        print_code(code, language)
                    elif code:
                        print_colored(f"Code:\n{code}", PNGUIN_COLOR)
                    else:
                        print_colored(part, PNGUIN_COLOR)
        else:
            print_colored(response, PNGUIN_COLOR)

# -----------
# MAIN LOOP
# -----------

async def main():
    global automode, conversation_history

    # Hooded ASCII figure introduction in a dark color
    ascii_hooded_figure = r"""
           ______
        .-'      '-.
       /            \
      |              |
      |,  .-.  .-.  ,|
      | )(_o/  \o_)( |
      |/     /\     \|
      (_     ^^     _)
       \__|IIIIII|__/
        | \IIIIII/ |
        \          /
         `--------`
       A hooded figure
        emerges...
    """
    print_colored(ascii_hooded_figure, HOODED_INTRO_COLOR)

    print_colored("Welcome to Pnguin Chat with Image Support!", PNGUIN_COLOR)
    print_colored("Slight flickers of cryptic code hum in the darkness.", HOODED_INTRO_COLOR)
    print_colored("Type 'exit' to end the conversation.", PNGUIN_COLOR)
    print_colored("Type 'image' to include an image in your message.", PNGUIN_COLOR)
    print_colored("Type 'automode [number]' to enter autonomous mode with a max iteration count.", PNGUIN_COLOR)

    while True:
        try:
            user_input = input(f"\n{USER_COLOR}You: {Style.RESET_ALL}")
        except KeyboardInterrupt:
            print_colored("\nUser interrupted. Exiting...", PNGUIN_COLOR)
            break

        if user_input.lower() == 'exit':
            print_colored("The hooded figure nods once and fades from view. Goodbye!", PNGUIN_COLOR)
            break

        if user_input.lower() == 'image':
            image_path = input(f"{USER_COLOR}Drag and drop your image path here: {Style.RESET_ALL}").strip().replace("'", "")
            if os.path.isfile(image_path):
                user_msg = input(f"{USER_COLOR}You (prompt for image): {Style.RESET_ALL}")
                response, _ = await chat_with_pnguin(user_msg, image_path)
                process_and_display_response(response)
            else:
                print_colored("Invalid image path. The terminal cracks with a hollow beep.", PNGUIN_COLOR)
            continue

        if user_input.lower().startswith('automode'):
            try:
                parts = user_input.split()
                if len(parts) > 1 and parts[1].isdigit():
                    max_iterations = int(parts[1])
                else:
                    max_iterations = MAX_CONTINUATION_ITERATIONS

                automode = True
                print_colored(f"Entering automode with {max_iterations} iterations...", TOOL_COLOR)
                print_colored("Press Ctrl+C at any time to exit automode prematurely.", TOOL_COLOR)

                # Get an initial user prompt outside the loop
                user_input = input(f"\n{USER_COLOR}You: {Style.RESET_ALL}")
                iteration_count = 0

                try:
                    while automode and iteration_count < max_iterations:
                        response, exit_continuation = await chat_with_pnguin(
                            user_input,
                            current_iteration=iteration_count + 1,
                            max_iterations=max_iterations
                        )
                        process_and_display_response(response)

                        if exit_continuation or CONTINUATION_EXIT_PHRASE in response:
                            print_colored("Automode completed. The hooded figure's presence wanes.", TOOL_COLOR)
                            automode = False
                        else:
                            print_colored(f"Continuation iteration {iteration_count + 1} complete.", TOOL_COLOR)
                            user_input = "Continue with the next step."

                        iteration_count += 1

                        if iteration_count >= max_iterations:
                            print_colored("Max iterations reached. Automode ends in a swirl of shadows.", TOOL_COLOR)
                            automode = False

                except KeyboardInterrupt:
                    print_colored("\nAutomode interrupted by user. The hooded figure reverts to manual listening.", TOOL_COLOR)
                    automode = False
                print_colored("Exited automode. Returning to the manual chat environment.", PNGUIN_COLOR)

            except KeyboardInterrupt:
                print_colored("\nAutomode manually disrupted. The figure stares, waiting for your next request.", TOOL_COLOR)
                automode = False

            continue
        else:
            response, _ = await chat_with_pnguin(user_input)
            process_and_display_response(response)

if __name__ == "__main__":
    asyncio.run(main())

