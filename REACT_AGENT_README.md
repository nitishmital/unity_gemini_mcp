# ReAct Agent for Unity Scene Manipulation

This implementation adds a **ReAct-style reasoning agent** to your existing Unity MCP system. The ReAct agent implements an **Observation → Reflection → Action → Repeat** loop that enables intelligent, goal-seeking behavior for Unity scene manipulation.

## 🎯 What is ReAct?

**ReAct** (Reasoning + Acting) is an AI agent architecture that combines:
- **Reasoning**: LLM-based planning and reflection
- **Acting**: Tool execution and environment interaction
- **Memory**: Persistent history of actions and outcomes
- **Visual Feedback**: Scene observation through rendering

This creates a **closed-loop interactive agent** similar to AutoGPT, Open Interpreter, or Voyager for Minecraft.

## 🚀 Key Features

### ✅ Core Components

1. **Goal-Seeking Agent**: Takes natural language goals and works toward completion
2. **Visual Observation**: Renders Unity scenes and analyzes them with Gemini Vision
3. **Memory System**: Maintains history of thoughts, actions, and reflections
4. **Intelligent Retries**: Reflects on failures and tries alternative approaches
5. **MCP Integration**: Uses your existing Unity MCP tools seamlessly

### ✅ ReAct Loop Structure

```
1. OBSERVE: Render scene → Analyze with vision
2. REASON: LLM plans next action based on goal and history
3. ACT: Execute MCP tool calls to Unity
4. REFLECT: Analyze results and plan next steps
5. REPEAT: Until goal achieved or max steps reached
```

## 📁 Files Overview

- `mcp_core.py` - Enhanced with `ReActAgent` class
- `test_react_agent.py` - Simple test script
- `react_agent_examples.py` - Comprehensive examples
- `REACT_AGENT_README.md` - This documentation

## 🛠️ Installation & Setup

### Prerequisites

1. **Google API Key**: Set your `GOOGLE_API_KEY` environment variable
2. **Unity MCP Server**: Ensure `unity_mcp_server.py` is running
3. **Python Dependencies**: Install from `requirements.txt`

```bash
export GOOGLE_API_KEY="your-api-key-here"
pip install -r requirements.txt
```

### Quick Start

```bash
# Test the ReAct agent
python test_react_agent.py

# Run examples
python react_agent_examples.py

# Interactive mode
python react_agent_examples.py interactive
```

## 🎮 Usage Examples

### Basic Usage

```python
from mcp_core import MCPCore

async def main():
    mcp_core = MCPCore()
    await mcp_core.connect("./unity_mcp_server.py")
    
    # Run ReAct agent with a goal
    result = await mcp_core.run_react_agent("Create a red cube on a green platform")
    print(result)
    
    await mcp_core.cleanup()

asyncio.run(main())
```

### Example Goals

1. **Simple Observation**: `"Observe the current Unity scene and describe what you see"`
2. **Object Creation**: `"Create a red cube at position (0, 1, 0) and a blue sphere at position (2, 0, 0)"`
3. **Object Manipulation**: `"Find the red cube and move it to position (0, 2, 0), then rotate it 45 degrees"`
4. **Complex Setup**: `"Create a scene with a green platform and place a red cube on top of it"`
5. **Physics**: `"Create a scene with a ground plane and a ball, then enable physics simulation"`

## 🧠 How It Works

### 1. Observation Phase
- Renders current Unity scene to PNG
- Uses Gemini Vision to analyze scene content
- Provides detailed description of objects, positions, and states

### 2. Reasoning Phase
- LLM analyzes goal vs current state
- Considers previous actions and their results
- Plans next action using available MCP tools

### 3. Action Phase
- Executes MCP tool calls to Unity
- Handles function calls and tool responses
- Captures results for reflection

### 4. Reflection Phase
- Analyzes action results
- Determines if goal is closer or further
- Plans alternative approaches if needed

### 5. Memory Management
- Stores complete history of thoughts, actions, results
- Uses memory to inform future decisions
- Prevents repeating failed approaches

## 🔧 Configuration

### ReActAgent Parameters

```python
react_agent = ReActAgent(mcp_core)
react_agent.max_steps = 20          # Maximum iterations
react_agent.memory = []             # Action history
react_agent.goal = "your goal"      # Current objective
```

### Custom Prompts

You can customize the reasoning prompts by modifying:
- `create_react_prompt()` - Main reasoning prompt
- `reflect_on_result()` - Reflection prompt
- `observe_scene()` - Scene analysis prompt

## 📊 Memory Structure

Each memory entry contains:
```json
{
  "step": 1,
  "thought": "I need to create a red cube",
  "action": "manage_gameobject with create parameters",
  "result": "Cube created successfully",
  "reflection": "The action worked as expected",
  "timestamp": "2024-01-01T12:00:00"
}
```

## 🎯 Goal Completion

The agent uses **visual verification** to determine goal completion:
- Renders scene after each action
- Uses Gemini Vision to analyze if goal is achieved
- Returns `GOAL_ACHIEVED`, `GOAL_NOT_ACHIEVED`, or `GOAL_PARTIAL`

## 🔍 Debugging & Monitoring

### Verbose Output
The agent provides detailed logging:
```
🔄 Step 1/20
👁️  Observing scene...
🧠 Reasoning and planning action...
💭 Thought: I need to create a red cube
⚡ Action: manage_gameobject with create parameters
🚀 Executing action...
📊 Result: Cube created successfully
🤔 Reflecting on result...
💡 Reflection: The action worked as expected
🎯 Checking goal completion...
```

### Error Handling
- Graceful handling of MCP tool failures
- Automatic retry with alternative approaches
- Detailed error messages for debugging

## 🚀 Advanced Features

### Custom Tool Integration
Add new MCP tools by extending the `execute_action()` method:

```python
async def execute_action(self, response):
    # Parse function calls
    for candidate in response.candidates:
        for part in candidate.content.parts:
            if hasattr(part, 'function_call'):
                # Execute custom tool logic
                pass
```

### Enhanced Memory
Implement more sophisticated memory management:
- Semantic similarity for action retrieval
- Long-term vs short-term memory
- Memory compression and summarization

### Multi-Modal Reasoning
Extend with additional observation types:
- Unity state data (object positions, properties)
- Audio feedback
- Performance metrics

## 🎯 Best Practices

### Writing Good Goals
- **Be specific**: "Create a red cube at (0,1,0)" vs "Make something red"
- **Break down complex tasks**: "First create platform, then place cube on top"
- **Include constraints**: "Use only primitive shapes" or "Keep objects within bounds"

### Monitoring Performance
- Watch the step count - high counts may indicate inefficient approaches
- Check memory entries for repeated failed actions
- Analyze reflection outputs for insight into agent reasoning

### Troubleshooting
- **Agent gets stuck**: Check if goal is too complex or ambiguous
- **Tool failures**: Verify MCP server connection and tool availability
- **Vision issues**: Ensure scene rendering is working properly

## 🔮 Future Enhancements

### Planned Features
1. **Hierarchical Planning**: Break complex goals into sub-goals
2. **Learning from Experience**: Improve performance over multiple runs
3. **Multi-Agent Coordination**: Multiple agents working together
4. **Real-time Adaptation**: Dynamic goal modification during execution

### Research Integration
- **Reflexion**: Enhanced reflection mechanisms
- **Chain-of-Thought**: More detailed reasoning traces
- **Self-Critique**: Agent evaluates its own performance

## 📚 References

- [ReAct Paper](https://arxiv.org/abs/2210.03629) - Original ReAct architecture
- [Reflexion](https://arxiv.org/abs/2303.11366) - Reflection mechanisms
- [AutoGPT](https://github.com/Significant-Gravitas/Auto-GPT) - Similar goal-seeking agents
- [Voyager](https://github.com/MineDojo/Voyager) - Minecraft agent inspiration

## 🤝 Contributing

To extend the ReAct agent:

1. **Add new observation types** in `observe_scene()`
2. **Enhance reasoning prompts** in `create_react_prompt()`
3. **Implement new tools** in `execute_action()`
4. **Improve goal checking** in `check_goal_completion()`

## 📄 License

This implementation builds on your existing MCP core system and follows the same licensing terms.

---

**Happy ReAct-ing! 🚀**

The ReAct agent transforms your Unity MCP system into an intelligent, goal-seeking assistant that can tackle complex scene manipulation tasks through systematic reasoning and visual feedback. 