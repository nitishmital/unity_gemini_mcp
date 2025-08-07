import asyncio
import google.generativeai as genai
from google.generativeai import types, GenerationConfig
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack
from typing import Optional, List, Dict, Any
import os
from dotenv import load_dotenv
import proto
import base64
from PIL import Image
import io
import json
from datetime import datetime

load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

def remove_key(container, key):
    if type(container) is dict:
        if key in container:
            del container[key]
        for v in container.values():
            remove_key(v, key)
    if type(container) is list:
        for v in container:
            remove_key(v, key)

class ReActAgent:
    """
    ReAct-style reasoning agent for Unity scene manipulation.
    Implements Observation -> Reflection -> Action -> Repeat loop.
    """
    
    TOOL_SCHEMAS = """
    MCP Tool Schemas:
    - manage_gameobject: {"action": "create|find|add_component|modify", "name": str, "position": {"x": float, "y": float, "z": float}, ...}
    - manage_editor: {"action": "play|stop|pause|resume"}
    - execute_menu_item: {"menu_path": str}
    - read_console: {"action": "read|clear", ...}
    - manage_scene: {"action": "open|save|new|close", ...}
    - manage_script: {"action": "create|edit|delete", ...}
    - manage_asset: {"action": "create|find|delete", ...}
    """
    
    def __init__(self, mcp_core: 'MCPCore'):
        self.mcp_core = mcp_core
        self.memory = []
        self.max_steps = 20
        self.current_step = 0
        self.goal = ""
        self.vision_model = genai.GenerativeModel('models/gemini-2.0-flash')
        self.reasoning_model = genai.GenerativeModel('models/gemini-2.5-flash')
        self.function_call_model = genai.GenerativeModel('models/gemini-2.5-flash')  # Dedicated model for function call generation
        self.last_scene_path = None
        
    def create_react_prompt(self, observation: str, last_action: Optional[Dict] = None) -> str:
        """Create a ReAct-style prompt with memory and current state."""
        memory_str = ""
        if self.memory:
            memory_str = "\n\nPrevious Actions:\n"
            for i, entry in enumerate(self.memory[-5:], 1):  # Last 5 actions
                memory_str += f"{i}. Thought: {entry.get('thought', 'N/A')}\n"
                memory_str += f"   Action: {entry.get('action', 'N/A')}\n"
                memory_str += f"   Result: {entry.get('result', 'N/A')}\n"
                if entry.get('reflection'):
                    memory_str += f"   Reflection: {entry['reflection']}\n"
                memory_str += "\n"
        last_action_str = ""
        if last_action:
            last_action_str = f"\nLast Action: {last_action.get('action', 'N/A')}\nLast Result: {last_action.get('result', 'N/A')}\n"
        
        # Build the prompt using string concatenation to avoid formatting issues
        #print("Tools: ", self.mcp_core.tools)
        tools_str = str(self.mcp_core.tools) if hasattr(self.mcp_core, 'tools') else "No tools available"
        
        prompt = f'''You are a ReAct-style reasoning agent for Unity scene manipulation. Your goal is to: {self.goal}

Current Observation: {observation}
{last_action_str}{memory_str}

{self.TOOL_SCHEMAS}

IMPORTANT: You MUST respond in this EXACT format:

Thought: [Your reasoning about what needs to be done, what went wrong, or what to try next]
Action: [The specific MCP tool call you want to make - use the available tools and schemas above] (Generate a function call here along with the text)
Observation: [Wait for the result]

Note that in your previous runs, you have used the wrong path to search for or create gameobjects or components. You used the 
path Assets/ , but that actually takes you to the path Assets/Assets/ since you are already within the root Assets/ directory.
Therefore, use the path / instead.
Available MCP Tools: {tools_str}

Guidelines:
- ALWAYS start with "Thought:" to explain your reasoning
- ALWAYS include "Action:" with the specific tool call you want to make
- Use precise coordinates and object names
- If an action fails, reflect on why and try a different approach
- If the goal appears achieved, use vision to verify
- Be systematic and methodical in your approach
- Consider the full context of previous actions and their results

EXAMPLE RESPONSE FORMAT:
Thought: I need to create a red cube in the scene. Looking at the current observation, I can see the scene is empty, so I should start by creating a cube GameObject.
Action: manage_gameobject(action="create", name="RedCube", position={{"x": 0, "y": 0.5, "z": 0}}) 
Function call: (Generate the function call to implement the above Action)
Observation: [Wait for the result]

Begin your reasoning:'''
        return prompt
    
    async def observe_scene(self, step=0) -> str:
        """Capture current scene state through rendering and analysis."""
        try:
            # Render current scene
            scene_path = f"/Users/nmital/Unity projects/MCP_builder/step_{self.current_step}_scene.png"
            self.last_scene_path = scene_path
            success = await self.mcp_core.render_scene(scene_path)
            if not success:
                return "Failed to render scene - cannot observe current state"
            # Analyze the scene using vision
            with open(scene_path, "rb") as img_file:
                img_data = img_file.read()
            img_part = {"mime_type": "image/png", "data": base64.b64encode(img_data).decode()}
            analysis_prompt = f"""Analyze this Unity scene image and describe what you see.

Goal: {self.goal}

Please describe:
1. What objects are visible in the scene?
2. What are their positions, colors, and states?
3. How does this relate to the goal: {self.goal}?
4. What needs to be changed to achieve the goal?

Provide a clear, detailed observation that will help plan the next action."""
            response = self.vision_model.generate_content([analysis_prompt, img_part])
            return response.text
        except Exception as e:
            return f"Error observing scene: {str(e)}"
    
    async def reflect_on_result(self, action: str, result: str, observation: str) -> str:
        """Use LLM to reflect on the action result and plan next steps."""
        reflection_prompt = f"""Reflect on the recent action and its result:

Goal: {self.goal}
Action taken: {action}
Result: {result}
Current observation: {observation}

Previous actions: {json.dumps(self.memory[-3:], indent=2) if self.memory else "None"}

Please analyze:
1. Did the action work as expected?
2. If not, what went wrong?
3. What should be tried next?
4. Are we closer to or further from the goal?
5. Should we try a different approach?

Provide a brief reflection that will guide the next action:"""
        response = self.reasoning_model.generate_content(reflection_prompt)
        return response.text
    
    async def check_goal_completion(self, observation: str) -> bool:
        """Check if the goal has been achieved using vision analysis."""
        try:
            scene_path = self.last_scene_path or self.mcp_core.modified_scene_path
            if not os.path.exists(scene_path):
                return False
            with open(scene_path, "rb") as img_file:
                img_data = img_file.read()
            img_part = {"mime_type": "image/png", "data": base64.b64encode(img_data).decode()}
            goal_check_prompt = f"""Analyze this Unity scene and determine if the goal has been achieved.

Goal: {self.goal}

Question: Does this scene show that the goal has been successfully completed?

Please respond with:
- "GOAL_ACHIEVED" if the goal is clearly completed
- "GOAL_NOT_ACHIEVED" if the goal is not yet completed
- "GOAL_PARTIAL" if there's partial progress

Provide a brief explanation for your assessment."""
            response = self.vision_model.generate_content([goal_check_prompt, img_part])
            result = response.text.strip()
            return "GOAL_ACHIEVED" in result.upper()
        except Exception as e:
            print(f"Error checking goal completion: {e}")
            return False
    
    async def execute_action(self, response: Any) -> Dict[str, Any]:
        """Execute an action using MCP tools and return the result."""
        try:
            found_function_call = False
            
            # Safely access response candidates
            candidates = getattr(response, 'candidates', [])
            if not candidates:
                print("⚠️  No candidates found in response")
                return {"success": False, "result": "No response candidates found"}
            
            for candidate in candidates:
                # Safely access content parts
                content = getattr(candidate, 'content', None)
                if not content:
                    continue
                
                parts = getattr(content, 'parts', [])
                if not parts:
                    continue
                
                for part in parts:
                    try:
                        # Check if part has function_call attribute
                        if not hasattr(part, 'function_call'):
                            continue
                        
                        # Safely check if function_call exists and is not None
                        function_call = getattr(part, 'function_call', None)
                        if not function_call:
                            continue
                        
                        found_function_call = True
                        
                        # Safely convert function call to dict
                        try:
                            function = proto.Message.to_dict(function_call)
                        except Exception as proto_error:
                            print(f"⚠️  Error converting function call to dict: {proto_error}")
                            # Try alternative approach
                            try:
                                function = {
                                    "name": getattr(function_call, 'name', 'unknown'),
                                    "args": getattr(function_call, 'args', {})
                                }
                            except Exception as alt_error:
                                print(f"❌ Alternative function call parsing failed: {alt_error}")
                                continue
                        
                        tool_name = function.get("name", "unknown")
                        tool_args = function.get("args", {})
                        
                        # Enhanced logging for tool usage
                        print(f"\n🔧 EXECUTING MCP TOOL:")
                        print(f"   Tool Name: {tool_name}")
                        print(f"   Parameters: {json.dumps(tool_args, indent=6)}")
                        
                        try:
                            tool_response = await self.mcp_core.session.call_tool(tool_name, tool_args)
                            if tool_response and tool_response.content:
                                response_text = tool_response.content[0].text
                                print(f"   ✅ Tool Response: {response_text}")
                                return {"success": True, "result": f"MCP Tool {tool_name}: {response_text}"}
                            else:
                                print(f"   ⚠️  MCP Tool {tool_name}: No response")
                                return {"success": False, "result": f"MCP Tool {tool_name}: No response"}
                        except Exception as e:
                            print(f"   ❌ MCP Tool {tool_name} error: {str(e)}")
                            return {"success": False, "result": f"MCP Tool {tool_name} error: {str(e)}"}
                    
                    except Exception as part_error:
                        print(f"⚠️  Error processing response part: {part_error}")
                        continue
        
        except Exception as e:
            print(f"❌ Error executing action: {str(e)}")
            return {"success": False, "result": f"Error executing action: {str(e)}"}
    
    async def _handle_response(self, response) -> str:
        """Handle LLM response with safe parsing and tool execution."""
        try:
            # Extract text content from response
            text_content = self._extract_response_text(response)
            
            # If no text content, return error
            '''if not text_content or text_content == "No text content found in response":
                return "No valid response generated" '''
            
            # Check for function calls in the response
            candidates = getattr(response, 'candidates', [])
            function_call_found = False
            
            if candidates:
                candidate = candidates[0]
                content = getattr(candidate, 'content', None)
                if content:
                    parts = getattr(content, 'parts', [])
                    for part in parts:
                        # Handle function call parts
                        if hasattr(part, 'function_call') and part.function_call:
                            function_call_found = True
                            try:
                                # Extract function call details
                                function = proto.Message.to_dict(part.function_call)
                                tool_name = function.get("name", "unknown")
                                tool_args = function.get("args", {})
                            except Exception:
                                # Fallback to direct attribute access
                                tool_name = getattr(part.function_call, 'name', 'unknown')
                                tool_args = getattr(part.function_call, 'args', {})
                            
                            # Execute the tool
                            try:
                                tool_response = await self.mcp_core.session.call_tool(tool_name, tool_args)
                                if tool_response and tool_response.content:
                                    response_text = tool_response.content[0].text
                                    return f"{text_content}\n\nTool executed: {tool_name}\nTool response: {response_text}"
                                else:
                                    return f"{text_content}\n\nTool executed: {tool_name}\nTool response: No response"
                            except Exception as e:
                                return f"{text_content}\n\nTool execution error: {str(e)}"
            if not function_call_found:
                action = await self._convert_text_action_to_function_call(text_content)
                action_response = await self.execute_action(action)
            
            # If no function calls and no parsable action, just return the text content
            return text_content
            
        except Exception as e:
            return f"Error handling response: {str(e)}"
    
    def _extract_response_text(self, response) -> str:
        """Safely extract text content from LLM response."""
        try:
            # Check for safety ratings first
            candidates = getattr(response, 'candidates', [])
            if not candidates:
                return "No text content found in response"
            
            candidate = candidates[0]
            safety_ratings = getattr(candidate, 'safety_ratings', [])
            if safety_ratings:
                for rating in safety_ratings:
                    if hasattr(rating, 'blocked') and rating.blocked:
                        return "Response blocked due to safety rating"
            
            # Try to extract text content
            content = getattr(candidate, 'content', None)
            if not content:
                return "No text content found in response"
            
            parts = getattr(content, 'parts', [])
            if not parts:
                return "No text content found in response"
            
            text_parts = []
            for part in parts:
                if hasattr(part, 'text') and part.text:
                    text_parts.append(part.text)
            
            if text_parts:
                return "\n".join(text_parts)
            
            # Fallback: try direct text access
            if hasattr(response, 'text'):
                return response.text
            
            return "No text content found in response"
            
        except Exception as e:
            return "Error extracting response text"
    
    async def _parse_action_from_text(self, text_content: str):
        """Parse action from text content and convert to function call format."""
        try:
            # Look for Action: line in the text
            lines = text_content.split('\n')
            action_line = None
            
            for line in lines:
                if line.strip().startswith('Action:'):
                    action_line = line.strip().replace('Action:', '').strip()
                    break
            
            if not action_line:
                return None
            
            print(f"\n🔍 PARSING ACTION FROM TEXT: {action_line}")
            
            # Use the dedicated function call model to convert text action to function call
            function_call = await self._convert_text_action_to_function_call(action_line)
            if function_call:
                return function_call
            
        except Exception as e:
            print(f"Error parsing action from text: {e}")
            return None
    
    async def _convert_text_action_to_function_call(self, action_text: str):
        """Convert text action to function call using dedicated Gemini model."""
        try:
            import re
            import json
            
            # Create a prompt specifically for function call generation
            function_call_prompt = f"""You are a function call generator for Unity MCP tools. Your task is to convert a text action into a proper function call.

Available MCP Tools and their schemas:
{self.TOOL_SCHEMAS}

Text Action: {action_text}

Instructions:
1. Analyze the text action and identify which MCP tool should be called
2. Extract the parameters from the text action
3. Generate a proper function call with the correct tool name and parameters
4. Return ONLY the function call object

IMPORTANT: Return ONLY the function call, no additional text or explanations.

Generate the function call for the above text action:"""

            # Generate function call using the dedicated model
            response = self.function_call_model.generate_content(
                contents=function_call_prompt,
                tools=self.mcp_core.available_tools,
                generation_config=GenerationConfig(
                    temperature=0.3,  # Low temperature for consistent output
                    max_output_tokens=500,
                    top_p=0.95,
                    top_k=40
                )
            )
            return response
        except Exception as e:
            print(f"Error converting to function call: {e}")
            return None
    
    async def run_react_loop(self, goal: str) -> str:
        self.goal = goal
        self.current_step = 0
        self.memory = []
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        print(f"🎯 Starting ReAct agent with goal: {goal}")
        print("=" * 80)
        # 1. OBSERVE
        print("👁️  OBSERVING ORIGINAL SCENE...")
        original_observation = await self.observe_scene(step=self.current_step)
        print(f"📋 Original Observation: {original_observation}")
        observation = original_observation
        while self.current_step < self.max_steps:
            self.current_step += 1
            print(f"\n🔄 STEP {self.current_step}/{self.max_steps}")
            print("=" * 50)
            # 2. REASON & ACT
            print("\n🧠 REASONING AND PLANNING ACTION...")
            print(observation)
            prompt = self.create_react_prompt(observation)
            
            # Log the prompt being sent to the LLM
            print(f"\n📝 Sending prompt to LLM:")
            print("-" * 40)
            #print(prompt)
            print("-" * 40)
            
            try:
                response = self.reasoning_model.generate_content(
                    contents=prompt,
                    tools=self.mcp_core.available_tools,
                    generation_config=GenerationConfig(
                        temperature=0.3,
                        max_output_tokens=2000,
                        top_p=0.95,
                        top_k=40,
                        stop_sequences=["TASK_COMPLETE", "EXIT", "SUCCESS", "FINISHED"]
                    )
                )
                
                # 3. HANDLE RESPONSE using the _handle_response method
                print("\n🚀 HANDLING LLM RESPONSE...")
                result = await self._handle_response(response) #self._extract_response_text(response) #await self._handle_response(response)
                print(f"\n📊 ACTION RESULT: {result}")
                print("👁️  OBSERVING MODIFIED SCENE...")
                observation = await self.observe_scene(step=self.current_step+1)
                print(f"📋 Observation: {observation}")
                result = result + f"Observation of effect of action: {observation}"
                
                # Extract thought and action for memory (if available)
                thought = ""
                action = ""
                
                # Try to extract from the result first, then from the original response
                response_text = result if result and result != "No valid response generated" else self._extract_response_text(response)
                
                if response_text and response_text != "No text content found in response" and response_text != "No valid response generated":
                    # Parse the response text for thoughts and actions
                    lines = response_text.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line.startswith('Thought:'):
                            thought = line.replace('Thought:', '').strip()
                        elif line.startswith('Action:'):
                            action = line.replace('Action:', '').strip()
                        elif line.startswith('Reasoning:'):
                            # If no explicit thought found, use reasoning
                            if not thought:
                                thought = line.replace('Reasoning:', '').strip()
                
                print(f"\n💭 EXTRACTED REASONING:")
                print(f"   Thought: {thought}")
                print(f"   Action: {action}")
                
                # Check if task is complete
                if "TASK_COMPLETE" in result:
                    print("\n✅ TASK COMPLETE!")
                    print("=" * 80)
                    return f"Goal '{goal}' successfully completed in {self.current_step} steps!"
                
                # Error handling: if the action failed, log and continue
                '''if "error" in result.lower() or "failed" in result.lower() or "no response" in result.lower():
                    consecutive_failures += 1
                    print(f"\n⚠️  ACTION FAILED. Consecutive failures: {consecutive_failures}/{max_consecutive_failures}")
                    
                    if consecutive_failures >= max_consecutive_failures:
                        print(f"\n❌ MAXIMUM CONSECUTIVE FAILURES REACHED ({max_consecutive_failures}). Stopping execution.")
                        return f"❌ Goal '{goal}' failed due to {max_consecutive_failures} consecutive failures. Final memory: {json.dumps(self.memory, indent=2)}"
                else:
                    consecutive_failures = 0  # Reset on success'''
                
                # 4. REFLECT
                print("\n🤔 REFLECTING ON RESULT...")
                reflection = await self.reflect_on_result(action, result, observation)
                print(f"💡 Reflection: {reflection}")
                
                # 5. STORE IN MEMORY
                memory_entry = {
                    "step": self.current_step,
                    "Original observation": original_observation,
                    "thought": thought,
                    "action": action,
                    "result": result,
                    "reflection": reflection,
                    "timestamp": datetime.now().isoformat()
                }
                self.memory.append(memory_entry)
                
                # 6. CHECK GOAL COMPLETION
                print("\n🎯 CHECKING GOAL COMPLETION...")
                goal_achieved = await self.check_goal_completion(observation + "\n"+ reflection)
                
                if goal_achieved:
                    print("\n✅ GOAL ACHIEVED!")
                    print("=" * 80)
                    return f"Goal '{goal}' successfully completed in {self.current_step} steps!"
                
                print("\n⏭️  CONTINUING TO NEXT STEP...")
                print("=" * 80)
            
            except Exception as llm_error:
                consecutive_failures += 1
                print(f"\n❌ LLM ERROR: {llm_error}")
                print(f"Consecutive failures: {consecutive_failures}/{max_consecutive_failures}")
                
                if consecutive_failures >= max_consecutive_failures:
                    print(f"\n❌ MAXIMUM CONSECUTIVE FAILURES REACHED ({max_consecutive_failures}). Stopping execution.")
                    return f"❌ Goal '{goal}' failed due to {max_consecutive_failures} consecutive failures. Final memory: {json.dumps(self.memory, indent=2)}"
                
                # Continue to next step even after LLM error
                print("⏭️  Continuing to next step despite LLM error...")
        
        print(f"\n❌ Goal '{goal}' not achieved within {self.max_steps} steps.")
        print("📊 FINAL MEMORY:")
        print(json.dumps(self.memory, indent=2))
        return f"❌ Goal '{goal}' not achieved within {self.max_steps} steps. Final memory: {json.dumps(self.memory, indent=2)}"

class MCPCore:
    SYSTEM_PROMPT = """System prompt: You are an expert in using the Unity game engine with visual feedback capabilities. Your overall task is to answer the query or task specified by the user within the Unity game-engine. You will use a step-by-step approach with visual verification:

1. FIRST: Render the current scene to 'current_scene.png' using Unity's SceneRenderer
2. REASON: Analyze the user's request and plan the necessary modifications
3. EXECUTE: Use MCP tools to modify the Unity scene
4. RENDER: Render the modified scene to 'modified_scene.png' using Unity's SceneRenderer
5. COMPARE: Use Gemini Vision to compare the images and verify the changes
6. EVALUATE: Determine if the desired effect has been achieved
7. EXIT: If successful, output 'TASK_COMPLETE', otherwise continue with alternative approaches

SCENE RENDERING INSTRUCTIONS:
- To render a scene, you must:
  1. Create or find a GameObject with a SceneRenderer component
  2. Set the fileName property in the SceneRenderer component to the desired filename
  3. Enter Play mode using {`action`: `play`} parameter within manage_editor tool.
  4. Wait for the SceneRenderer to execute and save the image
  5. Exit Play mode using {`action`: `stop`} parameter within manage_editor tool.
- Use manage_gameobject tools to create/find SceneRenderer and set its properties
- Use execute_menu_item to enter/exit Play mode
- The rendered images will be saved to the specified filepaths

Follow these guidelines:
- Always render scenes before and after modifications for visual verification
- Use chain-of-thought reasoning to break down complex operations
- Verify each operation's success through visual comparison
- Use precise coordinates and rotations
- Handle errors gracefully and suggest alternatives
- Confirm final results match the user's request through visual evidence
- If visual comparison shows the desired changes, exit with 'TASK_COMPLETE'"""

    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.available_tools = []
        self.chat_history = []
        self.original_scene_path = "/Users/nmital/Unity projects/MCP_builder/current_scene.png"
        self.modified_scene_path = "/Users/nmital/Unity projects/MCP_builder/modified_scene.png"
        self.vision_model = genai.GenerativeModel('models/gemini-2.0-flash-exp')

    async def connect(self, server_path: str) -> bool:
        try:
            command = "python" if server_path.endswith('.py') else "node"
            params = StdioServerParameters(command=command, args=[server_path], env=None)
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(params))
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
            await self.session.initialize()
            # Cache available tools
            response = await self.session.list_tools()
            self.tools = str(response)
            self.available_tools = self._process_tools(response.tools)
            return True
        except Exception as e:
            print(f"Connection error: {e}")
            return False

    def _process_tools(self, tools) -> List[types.Tool]:
        self._clean_schema(tools)
        return [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name=tool.name,
                        description=tool.description,
                        parameters=tool.inputSchema
                    )
                ]
            )
            for tool in tools
        ]

    def _clean_schema(self, tools):
        for tool in tools:
            remove_key(tool.inputSchema, "title")
            remove_key(tool.inputSchema, "default")
            remove_key(tool.inputSchema, "additionalProperties")

    async def render_scene(self, filename: str) -> bool:
        """Render the current Unity scene to a PNG file using Unity's SceneRenderer and manage_editor tool for Play mode."""
        try:
            # Step 1: Ensure SceneRenderer GameObject exists and is properly configured
            print("Setting up SceneRenderer for rendering...")
            # Try to find SceneRenderer
            find_response = await self.session.call_tool("manage_gameobject", {
                "action": "find",
                "name": "SceneRenderer"
            })
            if not find_response or not find_response.content:
                # Create SceneRenderer if it doesn't exist
                print("Creating SceneRenderer GameObject...")
                create_response = await self.session.call_tool("manage_gameobject", {
                    "action": "create",
                    "name": "SceneRenderer",
                    "position": {"x": 0, "y": 0, "z": 0}
                })
                if create_response and create_response.content:
                    print("SceneRenderer GameObject created")
                    # Add SceneRenderer component
                    component_response = await self.session.call_tool("manage_gameobject", {
                        "action": "add_component",
                        "objectName": "SceneRenderer",
                        "componentType": "SceneRenderer"
                    })
                    if component_response and component_response.content:
                        print("SceneRenderer component added")
                    else:
                        print("Failed to add SceneRenderer component")
                else:
                    print("Failed to create SceneRenderer GameObject")
                    return False
            else:
                print("SceneRenderer GameObject found")
            
            # Step 2: Enter Play mode for rendering using manage_editor
            print("Entering Play mode for scene rendering (manage_editor)...")
            play_mode_response = await self.session.call_tool("manage_editor", {
                "action": "play"
            })
            
            # Wait for Play mode to fully activate and SceneRenderer to execute
            import asyncio
            await asyncio.sleep(8)  # Give time for Play mode and rendering
            
            # Step 3: Exit Play mode after rendering using manage_editor
            print("Exiting Play mode (manage_editor)...")
            stop_mode_response = await self.session.call_tool("manage_editor", {
                "action": "stop"
            })
            
            # Step 4: Find the most recent PNG file and rename it to the target filename
            import os
            import shutil
            
            # Look for PNG files in the MCP_builder directory
            directory = "/Users/nmital/Unity projects/MCP_builder"
            if os.path.exists(directory):
                files = os.listdir(directory)
                png_files = [f for f in files if f.endswith('.png')]
                
                if png_files:
                    # Use the most recent PNG file
                    latest_file = max(png_files, key=lambda f: os.path.getmtime(os.path.join(directory, f)))
                    latest_path = os.path.join(directory, latest_file)
                    print(f"📁 Found latest PNG file: {latest_file}")
                    
                    # Copy the file to the target filename
                    try:
                        shutil.copy2(latest_path, filename)
                        print(f"✅ Copied {latest_file} to {filename}")
                        return True
                    except Exception as copy_error:
                        print(f"⚠️  Warning: Could not copy file to {filename}: {copy_error}")
                        return False
                else:
                    print(f"❌ No PNG files found in directory: {directory}")
                    return False
            else:
                print(f"❌ Directory not found: {directory}")
                return False
                
        except Exception as e:
            print(f"Error rendering scene: {e}")
            # Try to exit Play mode if we're still in it
            try:
                await self.session.call_tool("manage_editor", {
                    "action": "stop"
                })
            except:
                pass
            return False

    async def compare_scenes(self, original_path: str, modified_path: str, user_request: str) -> str:
        """Compare two scene images using direct file reading and Gemini Vision API"""
        try:
            import os
            import base64
            # Check if both image files exist
            if not os.path.exists(original_path):
                return f"Error: Original scene image not found: {original_path}"
            if not os.path.exists(modified_path):
                return f"Error: Modified scene image not found: {modified_path}"
            # Read image files directly using Python
            with open(original_path, "rb") as img1_file:
                img1_data = img1_file.read()
            with open(modified_path, "rb") as img2_file:
                img2_data = img2_file.read()
            # Create image parts for Gemini Vision API
            img1_part = {"mime_type": "image/png", "data": base64.b64encode(img1_data).decode()}
            img2_part = {"mime_type": "image/png", "data": base64.b64encode(img2_data).decode()}
            prompt = f"""Compare these two Unity scene images and analyze if the requested changes have been implemented.

User Request: {user_request}

Original Scene (left) vs Modified Scene (right)

Please analyze:
1. What visual differences do you see between the images?
2. Do these changes match what was requested by the user?
3. Are there any objects that were added, removed, moved, or modified?
4. Is the overall scene layout and composition as expected?
5. Are there any errors or unexpected changes?

Provide a detailed analysis and conclude with:
- "VISUAL_VERIFICATION_SUCCESS" if the changes match the request
- "VISUAL_VERIFICATION_FAILED" if the changes don't match or are incomplete
- "VISUAL_VERIFICATION_PARTIAL" if some changes are correct but incomplete"""
            response = self.vision_model.generate_content([prompt, img1_part, img2_part])
            return response.text
        except Exception as e:
            return f"Error comparing scenes: {str(e)}"

    async def cleanup(self):
        await self.exit_stack.aclose()

    async def run_react_agent(self, goal: str) -> str:
        """Run the ReAct agent with the given goal."""
        react_agent = ReActAgent(self)
        return await react_agent.run_react_loop(goal)


'''Example outputs:

Text Action: manage_gameobject(action="create", name="RedCube", position={{"x": 0, "y": 0.5, "z": 0}})
{{"name": "manage_gameobject", "args": {{"action": "create", "name": "RedCube", "position": {{"x": 0, "y": 0.5, "z": 0}}}}}}

Text Action: manage_gameobject(action="modify", target="Cube", component_properties={{"MeshRenderer": {{"material.color": [1, 0, 0, 1]}}}})
{{"name": "manage_gameobject", "args": {{"action": "modify", "name": "Cube", "component_properties": {{"MeshRenderer": {{"material.color": [1, 0, 0, 1]}}}}}}}}

Text Action: manage_editor(action="play")
{{"name": "manage_editor", "args": {{"action": "play"}}}}

Text Action: execute_menu_item(menu_path="GameObject/3D Object/Cube")
{{"name": "execute_menu_item", "args": {{"menu_path": "GameObject/3D Object/Cube"}}}}'''

# Try to parse the JSON response
'''try:
    # Look for JSON in the response
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        function_call = json.loads(json_str)

        # Validate the function call structure
        if "name" in function_call and "args" in function_call:
            print(f"✅ Successfully parsed function call: {function_call}")
            return function_call
        else:
            print(f"⚠️  Invalid function call structure: {function_call}")
            return None
    else:
        print(f"⚠️  No JSON found in response: {response_text}")
        return None

except json.JSONDecodeError as e:
    print(f"❌ Error parsing JSON from function call response: {e}")
    return None
except Exception as e:
    print(f"❌ Error processing function call response: {e}")
    return None

except Exception as e:
print(f"❌ Error converting text action to function call: {e}")
return None'''