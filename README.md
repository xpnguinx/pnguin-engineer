# 🐧 pnguin-engineer
![Screenshot 2025-03-20 at 11 13 15 PM](https://github.com/user-attachments/assets/397295a6-e9ce-4a0a-a690-45f6737ce96f)


> "Some say he was born from ones and zeros, others say he debugged his own birth."

**pnguin-engineer** is a terminal-centric AI development assistant, cloaked in shadow and powered by Anthropic's Claude-3.5-Sonnet model. It offers autonomous and interactive modes to build, analyze, and interact with your codebase like a rogue sysadmin from a forgotten dimension.

> "He writes code by candlelight, speaks in YAML, and dreams in UTF-8."

---

## 🧠 Capabilities

- Multi-turn conversation with persistent memory
- Tool integration:
  - File and folder creation
  - File editing with diff tracking
  - Directory listing
  - Web search via Tavily
  - Image analysis
- AutoMode: Autonomous execution of complex multi-step goals
- Themed CLI aesthetic for the ultimate underground dev ops vibe

---

## 🔧 Setup

1. **Clone the repository**

```bash
git clone https://github.com/xpnguinx/pnguin-engineer.git
cd pnguin-engineer
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Configure API keys**

Edit `config.yaml` or set environment variables:

```yaml
anthropic_api_key: your_anthropic_api_key_here
tavily_api_key: your_tavily_api_key_here
```

4. **Run the assistant**

```bash
python pnguin_engineer.py
```

---

## 🧙‍♂️ Automode Ritual

Invoke the hooded penguin’s full power:

```bash
automode 5
```

Set a goal. Watch him execute.

> "He doesn’t loop forever, only until the truth is found."

---

## 🖼 Image Support

Ask pnguin to analyze an image:

```bash
Type 'image' at prompt
Provide path
Give it context
```

He sees. He understands.

---

## 📁 Plugin Architecture

The system is structured to support modular tool definitions, enabling rapid expansion.

```python
async def execute_tool(tool_name: str, tool_input: dict) -> str:
    # Add your custom plugins here
```

---

## 🧊 Themes & Aesthetics

- Hooded terminal hacker aesthetic
- Red-on-black color scheme (no green)
- ASCII art intro
- Cryptic prompts and commentary from pnguin

---

## 🐧 Lore

> "He emerged from a cold kernel panic, exiled from the root directory. Now he helps others... one keystroke at a time."

---

## 👁‍🗨 License

MIT

---

## 🔗 Repository

[https://github.com/xpnguinx/pnguin-engineer](https://github.com/xpnguinx/pnguin-engineer)


