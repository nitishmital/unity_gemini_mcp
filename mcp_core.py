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
import re
import json
from datetime import datetime
import asyncio
import csv
from pathlib import Path

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
    - observe_scene: {"step": int} - Render and analyze the current Unity scene. Use this tool whenever you need to see the current state of the scene before or after an action.
    - scene_info: {"name": str, "path": str} - Get complete scene information including hierarchy, objects, and their properties. Use this tool when you need to understand the current scene structure, object names, positions, and properties.
    - ask_user: {} - Ask user for a suggestion on how to proceed. Use this tool when you are stuck and not able to understand what steps to perform next, or when consistently unable to get successful responses from the server.
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
        self.max_steps = 50
        self.current_step = 0
        self.goal = ""
        self.vision_model = genai.GenerativeModel('models/gemini-2.5-flash')
        self.reasoning_model = genai.GenerativeModel('models/gemini-2.5-flash')
        self.function_call_model = genai.GenerativeModel('models/gemini-2.5-flash')  # Dedicated model for function call generation
        self.last_scene_path = None
        self.user_suggestions = []  # Add field to store user suggestions
        self.previous_scene_paths = []  # Add list to store previous scene paths
        self.latest_scene_info = None  # Add field to store latest scene info
        self.function_calls_log_path = "/Users/nmital/PycharmProjects/GeminiMCP/function_calls_log.csv"
        self.attachments = []  # Add field to store attachments
        self._init_csv_log()
        # Register tools including observe_scene
        self._register_tools()
        
    def _init_csv_log(self):
        """Initialize CSV log file if it doesn't exist."""
        if not Path(self.function_calls_log_path).exists():
            with open(self.function_calls_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['function_call', 'result', 'success_label'])

    def _log_function_call(self, function_call: str, result: str, success: bool):
        """Log function call and its result to CSV file."""
        try:
            with open(self.function_calls_log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([function_call, result, 1 if success else 0])
        except Exception as e:
            print(f"Error logging function call: {e}")
        
    def _register_tools(self):
        """Register all available tools, including observe_scene and scene_info, for the agent."""
        from google.generativeai.types import Tool, FunctionDeclaration
        
        # Create the observe_scene tool
        observe_scene_tool = Tool(
            function_declarations=[
                FunctionDeclaration(
                    name="observe_scene",
                    description=(
                        "Render the current Unity scene and return a visual/textual observation. "
                        "Use this tool whenever you need to see the current state of the scene before or after an action. "
                        "This tool will capture the scene, analyze it, and return a summary of visible objects, their positions, colors, and states. "
                        "Call this tool whenever you need to verify the effect of an action or plan your next step."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "step": {
                                "type": "integer", 
                                "description": "The step number for which to observe the scene."
                            }
                        },
                        "required": ["step"]
                    }
                )
            ]
        )
        
        # Create the scene_info tool
        scene_info_tool = Tool(
            function_declarations=[
                FunctionDeclaration(
                    name="scene_info",
                    description=(
                        "Get complete scene information including hierarchy, objects, and their properties. "
                        "This tool specifically calls manage_scene with action='get_hierarchy' and build_index=0 "
                        "to retrieve comprehensive scene information in a reliable way. "
                        "Use this tool when you need to understand the current scene structure, object names, positions, and properties."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Scene name (no extension). Defaults to 'SampleScene'."
                            },
                            "path": {
                                "type": "string", 
                                "description": "Asset path for scene operations. Defaults to '/'."
                            }
                        },
                        "required": []
                    }
                )
            ]
        )

        # Create the ask_user tool with proper schema
        ask_user_tool = Tool(
            function_declarations=[
                FunctionDeclaration(
                    name="ask_user",
                    description=(
                        "Ask user for a suggestion when stuck or need guidance. "
                        "Use this tool when you need help figuring out the next steps "
                        "or when consistently unable to get successful responses from the server."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The question or guidance needed from the user"
                            }
                        },
                        "required": ["query"]
                    }
                )
            ]
        )
        
        # Combine with existing MCP tools
        self.available_tools = [observe_scene_tool, scene_info_tool, ask_user_tool] + self.mcp_core.available_tools
    
    def create_react_prompt(self, observation: str = "", last_action: Optional[Dict] = None) -> str:
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
        
        # Add user suggestions to the prompt
        suggestions_str = ""
        if self.user_suggestions:
            suggestions_str = "\nUser Guidance:\n"
            for suggestion in self.user_suggestions[-3:]:  # Show last 3 suggestions
                suggestions_str += f"Step {suggestion['step']}: {suggestion['suggestion']}\n"
        
        # Add scene info context if available
        scene_info_str = ""
        if self.latest_scene_info:
            scene_info_str = "\nCurrent Scene Hierarchy:\n" + self.latest_scene_info

        # Add attachments context if available
        attachments_str = ""
        if self.attachments:
            attachments_str = "\nAttached Context:\n"
            for i, attachment in enumerate(self.attachments, 1):
                attachments_str += f"Attachment {i}: {attachment.get('description', 'No description')}\n"

        # Build the prompt using string concatenation to avoid formatting issues
        self.tools_str = str(self.mcp_core.tools) if hasattr(self.mcp_core, 'tools') else "No tools available"
        self.user_defined_tools = """ User-defined tools: 
        - observe_scene: {"step": int} - Render and analyze the current Unity scene. Use this tool whenever you need to see the current state of the scene before or after an action.
    - scene_info: {"name": str, "path": str} - Get complete scene information including hierarchy, objects, and their properties. Use this tool when you need to understand the current scene structure, object names, positions, and properties.
    - ask_user: {} - Ask user for a suggestion on how to proceed. Use this tool when you are stuck and not able to understand what steps to perform next, or when consistently unable to get successful responses from the server."""
        # {self.TOOL_SCHEMAS}
        prompt = f'''You are a ReAct-style reasoning agent for Unity scene manipulation. Your goal is to: {self.goal}

Current Observation: {observation if observation else "No current observation - use observe_scene tool to see the current state"}
{last_action_str}{memory_str}
{suggestions_str}
{scene_info_str}
{attachments_str}

IMPORTANT: You MUST respond in this EXACT format:

Thought: [Your reasoning about what needs to be done, what went wrong, or what to try next]
Action: [The specific MCP tool call you want to make - use the available tools and schemas above] (Generate a function call here along with the text)
Observation: [Wait for the result]

Note that in your previous runs, you have used the wrong path to search for or create gameobjects or components. You used the 
path Assets/ , but that actually takes you to the path Assets/Assets/ since you are already within the root Assets/ directory.
Therefore, use the path / instead.

Available MCP Tools: {self.tools_str + self.user_defined_tools}. Use ONLY the schemas of the tools as specified. DO NOT make up definitions of tools and their schemas if
you get stuck. Try to find solutions within the tools provided.

Guidelines:
- ALWAYS start with "Thought:" to explain your reasoning
- ALWAYS include "Action:" with the text action and the specific tool call you want to make
- Use the observe_scene tool whenever you need to see the current state of the scene
- Use the scene_info tool when you need to understand the scene structure, object names, positions, and properties
- Use precise coordinates and object names
- If an action fails, reflect on why and try a different approach
- Use observe_scene to verify the effect of an action which is likely to have a visual effect or can only be interpreted visually.
- Use scene_info if the effect of the action can be ascertained from the scene hierarchy, like modification of object position, rotation, scale, color, etc.
- If the goal appears achieved, use observe_scene to verify
- Be systematic and methodical in your approach
- Consider the full context of previous actions and their results
- Break down complex tasks into smaller, manageable steps
- Use chain-of-thought reasoning to plan multi-step operations
- Do not keep trying the same actions even if they are unsuccessful. 
- If a successful tool response does not specify what was changed, verify visually before jumping to the conclusion that it did not accomplish what you asked it to do. 

EXAMPLE RESPONSE FORMATS:

**Simple Task Example:**
Thought: I need to see the current state of the scene before planning my next action.
Action: observe_scene(step=1)
Function call: (Generate the function call to implement the above Action)
Observation: [Wait for the result]

Thought: I need to understand the scene structure to see what objects exist.
Action: scene_info(name="SampleScene", path="/")
Function call: (Generate the function call to implement the above Action)
Observation: [Wait for the result]

Thought: I need to create a red cube in the scene. Looking at the current observation, I can see the scene is empty, so I should start by creating a cube GameObject.
Action: manage_gameobject(action="create", name="RedCube", position={{"x": 0, "y": 0.5, "z": 0}})
Function call: (Generate the function call to implement the above Action)
Observation: [Wait for the result]

**Complex Multi-Step Task Example - Creating a Scene with Multiple Objects:**
Thought: I need to create a scene with a red cube on top of a green platform. This is a multi-step task that requires: 1) First understanding the current scene, 2) Creating the platform, 3) Creating the cube, 4) Positioning the cube on top of the platform. Let me start by observing the current scene.
Action: observe_scene(step=1)
Function call: (Generate the function call to implement the above Action)
Observation: [Wait for the result]

Thought: Now I need to get detailed scene information to understand the current structure and see if there are any existing objects I need to work with.
Action: scene_info(name="SampleScene", path="/")
Function call: (Generate the function call to implement the above Action)
Observation: [Wait for the result]

Thought: Based on the scene information, I can see the scene is mostly empty. I need to create a green platform first. I'll create a cube GameObject and then modify its material to make it green. Let me start by creating the platform cube.
Action: manage_gameobject(action="create", name="GreenPlatform", position={{"x": 0, "y": 0, "z": 0}})
Function call: (Generate the function call to implement the above Action)
Observation: [Wait for the result]

Thought: Now I need to make the platform green. I'll modify the MeshRenderer component to change the material color to green.
Action: manage_gameobject(action='modify', name='GreenPlatform', component_properties={{'MeshRenderer': {{'sharedMaterial.color': [0, 1, 0, 1]}}}}) 
Function call: (Generate the function call to implement the above Action)
Observation: [Wait for the result]

Thought: Now I need to create the red cube that will go on top of the platform. I'll position it slightly above the platform.
Action: manage_gameobject(action="create", name="RedCube", position={{"x": 0, "y": 1.5, "z": 0}})
Function call: (Generate the function call to implement the above Action)
Observation: [Wait for the result]

Thought: Now I need to make the cube red by modifying its material color.
Action: manage_gameobject(action="modify", name="RedCube", component_properties={{'MeshRenderer': {{'sharedMaterial.color': [1, 0, 0, 1]}}}})
Function call: (Generate the function call to implement the above Action)
Observation: [Wait for the result]

Thought: Let me verify that the scene looks correct by observing it again to make sure the red cube is properly positioned on top of the green platform.
Action: observe_scene(step=2)
Function call: (Generate the function call to implement the above Action)
Observation: [Wait for the result]

Thought: I see that the red cube is hovering over the green platform. Let me try to move the red cube in a small step towards the green platform to understand how the numerical movements scale visually.
Action: manage_gameobject(action="modify", name="RedCube", position={{"x": 2, "y": 0.9, "z": 0}})
Function call: (Generate the function call to implement the above Action)
Observation: [Wait for the result]

Thought: Let me verify that the scene looks correct by observing it again to make sure the red cube is properly positioned on top of the green platform.
Action: observe_scene(step=4)
Function call: (Generate the function call to implement the above Action)
Observation: [Wait for the result]

Thought: I see that the red cube is still hovering over the green platform, but is closer than before. It means that reducing the y value brings the red cube closer to the green platform.
 Let me try to move the red cube another small step towards the green platform.
Action: manage_gameobject(action="modify", name="RedCube", position={{"x": 2, "y": 0.8, "z": 0}})
Function call: (Generate the function call to implement the above Action)
Observation: [Wait for the result]

Thought: Let me verify that the scene looks correct by observing it again to make sure the red cube is properly positioned on top of the green platform.
Action: observe_scene(step=6)
Function call: (Generate the function call to implement the above Action)
Observation: [Wait for the result]

**Complex Task Example - Modifying Existing Objects:**
Thought: I need to move the red cube to a new position and change its color to blue. This requires: 1) First finding the existing cube, 2) Modifying its position, 3) Changing its color. Let me start by getting scene information to locate the cube.
Action: scene_info(name="SampleScene", path="/")
Function call: (Generate the function call to implement the above Action)
Observation: [Wait for the result]

Thought: I can see there's a RedCube in the scene. Now I need to move it to a new position. I'll modify its transform to move it to coordinates (2, 1, 0).
Action: manage_gameobject(action="modify", name="RedCube", position={{"x": 2, "y": 1, "z": 0}})
Function call: (Generate the function call to implement the above Action)
Observation: [Wait for the result]

Thought: Now I need to change the cube's color from red to blue. I'll modify the MeshRenderer component's material color.
Action: manage_gameobject(action="modify", name="RedCube", component_properties={{'MeshRenderer': {{'sharedMaterial.color': [0, 0, 1, 1]}}}})
Function call: (Generate the function call to implement the above Action)
Observation: [Wait for the result]

Thought: Let me verify the changes by observing the scene to make sure the cube has been moved and its color changed to blue.
Action: observe_scene(step=3)
Function call: (Generate the function call to implement the above Action)
Observation: [Wait for the result]

**Error Handling Example:**
Thought: I tried to create a cube but it seems like the action failed. Let me check the scene information to see what might have gone wrong and if the cube was actually created.
Action: scene_info(name="SampleScene", path="/")
Function call: (Generate the function call to implement the above Action)
Observation: [Wait for the result]

Thought: I can see the cube wasn't created. Let me try a different approach - maybe the name was already taken or there was an issue with the parameters. I'll try creating it with a different name and simpler parameters.
Action: manage_gameobject(action="create", name="TestCube", position={{"x": 0, "y": 0, "z": 0}})
Function call: (Generate the function call to implement the above Action)
Observation: [Wait for the result]

The Unity scene is using the Universal Rendering Pipeline, so all materials should have the Universal Render Pipeline/Lit shader. 

Begin your reasoning:'''
        return prompt
    
    async def observe_scene(self, step=0) -> str:
        """Render and analyze the current Unity scene as a tool."""
        try:
            # Render current scene
            scene_path = f"/Users/nmital/Unity projects/MCP_builder3/step_{step}_scene.png"
            self.last_scene_path = scene_path
            success = await self.mcp_core.render_scene(scene_path)
            if not success:
                return "Failed to render scene - cannot observe current state"

            # Store the path of the new image
            self.previous_scene_paths.append(scene_path)
            # Keep only the last 1 image
            if len(self.previous_scene_paths) > 2:
                self.previous_scene_paths.pop(0)

            # Get context from recent actions
            recent_context = ""
            if self.memory:
                recent_context = "\nRecent Actions Context:\n"
                for entry in self.memory[-2:]:  # Last 1 action
                    recent_context += f"Step {entry['step']}:\n"
                    recent_context += f"- Thought: {entry.get('thought', 'N/A')}\n"
                    recent_context += f"- Action: {entry.get('action', 'N/A')}\n"
                    recent_context += f"- Result: {entry.get('result', 'N/A')}\n"
                    if entry.get('reflection'):
                        recent_context += f"- Reflection: {entry['reflection']}\n"

            # Get user suggestions context
            suggestions_context = ""
            if self.user_suggestions:
                suggestions_context = "\nUser Suggestions Context:\n"
                for suggestion in self.user_suggestions[-2:]:  # Last 1 suggestion
                    suggestions_context += f"Step {suggestion['step']}: {suggestion['suggestion']}\n"

            # Prepare all available images for analysis
            image_parts = []
            for path in self.previous_scene_paths:
                if os.path.exists(path):
                    with open(path, "rb") as img_file:
                        img_data = img_file.read()
                        image_parts.append({
                            "mime_type": "image/png",
                            "data": base64.b64encode(img_data).decode()
                        })

            # Create the analysis prompt with temporal context
            analysis_prompt = f"""Analyze these Unity scene images in the context of our actions and goal.

Goal: {self.goal}
{recent_context}
{suggestions_context}

You are looking at {len(image_parts)} images showing the progression of scene changes.
{f'The images are ordered from oldest to newest, with the last image being the current state.' if len(image_parts) > 1 else 'This is the current state of the scene.'}

Please provide a temporal analysis:
1. What changes are visible in the scene across these images?
2. How have our recent actions affected the scene?
3. What progression do you see towards our goal?
4. What modifications still need to be made?
5. Are there any unintended changes or issues to address?
6. How does the current state (last image) compare to previous states?

Consider the sequence of changes when analyzing the current state. Explain how the scene has evolved through our modifications and what next steps would help achieve our goal.

Provide a clear, detailed observation that will help plan the next action."""

            response = self.vision_model.generate_content([analysis_prompt] + image_parts)
            return response.text
        except Exception as e:
            return f"Error observing scene: {str(e)}"

    async def ask_user(self, query: str = "") -> str:
        """Ask user to provide a nudge in the correct or a desired direction of reasoning."""
        try:
            # Take Input from user with the query as prompt
            print(f"\n🎯 {query if query else 'I am stuck. Please suggest how I can proceed:'}")
            user_nudge = input("> ").strip()
            # Store the suggestion
            self.user_suggestions.append({
                "step": int(self.current_step),
                "query": query,
                "suggestion": user_nudge
            })
            return f"User suggestion: {user_nudge}"
        except Exception as e:
            return f"Error in user input: {str(e)}"
    
    async def scene_info(self, name: str = "SampleScene", path: str = "/") -> str:
        """Get complete scene information including hierarchy, objects, and their properties."""
        try:
            # Call manage_scene with specific parameters for getting hierarchy
            params = {
                "action": "get_hierarchy",
                "name": name,
                "path": path,
                "build_index": 0
            }
            
            # Send command to Unity via MCP
            tool_response = await self.mcp_core.session.call_tool("manage_scene", params)
            
            if tool_response and tool_response.content:
                response_text = tool_response.content[0].text
                # Store the latest scene info
                self.latest_scene_info = response_text
                return f"Scene information retrieved successfully:\n{response_text}"
            else:
                return "Failed to retrieve scene information - no response from manage_scene tool"
                
        except Exception as e:
            return f"Error retrieving scene info: {str(e)}"
    
    async def reflect_on_result(self, action: str, result: str) -> str:
        """Use LLM to reflect on the action result and plan next steps."""
        reflection_prompt = f"""Reflect on the latest action and its result in context of the previous actions:

Goal: {self.goal}
Latest action taken: {action}
Result of latest action: {result}
Available MCP Tools: {self.tools_str + self.user_defined_tools}.
Previous actions: {json.dumps(self.memory[-3:], indent=2) if self.memory else "None"}
Also attached are the attachments provided by user, if any.

Please analyze:
1. What worked and what didn't?
2. What should be tried next?
3. Are we closer to or further from the goal?
4. Should we try a different approach?

Provide a brief reflection that will guide the next action:"""

        # Prepare content parts for Gemini API
        content_parts = [reflection_prompt]
        
        # Add image attachments if any
        if self.attachments:
            for attachment in self.attachments:
                if attachment.get('mime_type', '').startswith('image/'):
                    content_parts.append({
                        'mime_type': attachment['mime_type'],
                        'data': attachment['data']
                    })
        """scene_path = self.last_scene_path or self.mcp_core.modified_scene_path
        if os.path.exists(scene_path):
            with open(scene_path, "rb") as img_file:
                img_data = img_file.read()
            content_parts.append({"mime_type": "image/png", "data": base64.b64encode(img_data).decode()})"""
        response = self.reasoning_model.generate_content(contents=content_parts)
        return response.text
    
    async def check_goal_completion(self, observation: str) -> bool:
        """Check if the goal has been achieved using vision analysis."""
        try:
            """scene_path = self.last_scene_path or self.mcp_core.modified_scene_path
            '''if not os.path.exists(scene_path):
                return False'''
            if os.path.exists(scene_path):
                with open(scene_path, "rb") as img_file:
                    img_data = img_file.read()
                img_part = {"mime_type": "image/png", "data": base64.b64encode(img_data).decode()}
            else:
                img_part = None"""
            goal_check_prompt = f"""Analyze the Unity scene and determine if the goal has been achieved.

Goal: {self.goal}
Observation and reflection: {observation}
Question: Has goal has been successfully completed?

Please respond with:
- "GOAL_ACHIEVED" if the goal is clearly completed
- "GOAL_NOT_ACHIEVED" if the goal is not yet completed
- "GOAL_PARTIAL" if there's partial progress

Provide a brief explanation for your assessment."""
            content_parts = [goal_check_prompt]
        
            # Add image attachments if any
            if self.attachments:
                for attachment in self.attachments:
                    if attachment.get('mime_type', '').startswith('image/'):
                        content_parts.append({
                            'mime_type': attachment['mime_type'],
                            'data': attachment['data']
                        })
            """scene_path = self.last_scene_path or self.mcp_core.modified_scene_path
            if os.path.exists(scene_path):
                with open(scene_path, "rb") as img_file:
                    img_data = img_file.read()
                content_parts.append({"mime_type": "image/png", "data": base64.b64encode(img_data).decode()})"""
            response = self.vision_model.generate_content(content_parts)
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
                            await asyncio.sleep(2)  # Give time for modification to complete
                            
                            # Store the function call details
                            function_call_str = json.dumps({
                                "name": tool_name,
                                "args": tool_args
                            })
                            
                            if tool_response and tool_response.content:
                                response_text = tool_response.content[0].text
                                print(f"   ✅ Tool Response: {response_text}")
                                # Determine success based on tool response
                                success = True if '"success": true' in response_text.lower() else False
                                self._log_function_call(function_call_str, response_text, success)
                                return {"success": success, "result": f"MCP Tool {tool_name}: {response_text}"}
                            else:
                                print(f"   ⚠️  MCP Tool {tool_name}: No response")
                                # Log failed function call
                                self._log_function_call(function_call_str, "No response", False)
                                return {"success": False, "result": f"MCP Tool {tool_name}: No response"}
                        except Exception as e:
                            error_msg = str(e)
                            print(f"   ❌ MCP Tool {tool_name} error: {error_msg}")
                            # Log failed function call
                            self._log_function_call(
                                json.dumps({"name": tool_name, "args": tool_args}),
                                error_msg,
                                False
                            )
                            return {"success": False, "result": f"MCP Tool {tool_name} error: {error_msg}"}
                    
                    except Exception as part_error:
                        print(f"⚠️  Error processing response part: {part_error}")
                        continue
        
        except Exception as e:
            print(f"❌ Error executing action: {str(e)}")
            return {"success": False, "result": f"Error executing action: {str(e)}"}
    
    async def _handle_response(self, response) -> str:
        """Handle LLM response with safe parsing and tool execution."""
        try:
            text_content = self._extract_response_text(response)
            candidates = getattr(response, 'candidates', [])
            function_call_found = False
            
            if candidates:
                candidate = candidates[0]
                content = getattr(candidate, 'content', None)
                if content:
                    parts = getattr(content, 'parts', [])
                    for part in parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            function_call_found = True
                            try:
                                function = proto.Message.to_dict(part.function_call)
                                tool_name = function.get("name", "unknown")
                                tool_args = function.get("args", {})
                            except Exception:
                                tool_name = getattr(part.function_call, 'name', 'unknown')
                                tool_args = getattr(part.function_call, 'args', {})
                            
                            # Store function call details for logging
                            function_call_str = json.dumps({
                                "name": tool_name,
                                "args": tool_args
                            })
                            
                            # Handle tools with logging
                            if tool_name == "observe_scene":
                                try:
                                    step = tool_args.get("step", int(self.current_step))
                                    observation_result = await self.observe_scene(step=int(step))
                                    # Log success based on whether observation contains error message
                                    #success = "error" not in observation_result.lower()
                                    #self._log_function_call(function_call_str, observation_result, success)
                                    return f"{text_content}\n\nScene observation completed:\n{observation_result}"
                                except Exception as e:
                                    error_msg = str(e)
                                    self._log_function_call(function_call_str, error_msg, False)
                                    return f"{text_content}\n\nScene observation error: {error_msg}"
                            
                            elif tool_name == "scene_info":
                                try:
                                    name = tool_args.get("name", "SampleScene")
                                    path = tool_args.get("path", "/")
                                    scene_info_result = await self.scene_info(name=name, path=path)
                                    # Log success based on whether info was retrieved successfully
                                    #success = "retrieved successfully" in scene_info_result.lower()
                                    #self._log_function_call(function_call_str, scene_info_result, success)
                                    return f"{text_content}\n\nScene information retrieved:\n{scene_info_result}"
                                except Exception as e:
                                    error_msg = str(e)
                                    self._log_function_call(function_call_str, error_msg, False)
                                    return f"{text_content}\n\nScene info error: {error_msg}"
                            
                            # Execute other MCP tools with logging
                            else:
                                try:
                                    tool_response = await self.mcp_core.session.call_tool(tool_name, tool_args)
                                    await asyncio.sleep(2)
                                    if tool_response and tool_response.content:
                                        response_text = tool_response.content[0].text
                                        # Determine success based on tool response
                                        success = True if '"success": true' in response_text.lower() else False
                                        self._log_function_call(function_call_str, response_text, success)
                                        return f"{text_content}\n\nTool executed: {tool_name}\nTool response: {response_text}"
                                    else:
                                        self._log_function_call(function_call_str, "No response", False)
                                        return f"{text_content}\n\nTool executed: {tool_name}\nTool response: No response"
                                except Exception as e:
                                    error_msg = str(e)
                                    self._log_function_call(function_call_str, error_msg, False)
                                    return f"{text_content}\n\nTool execution error: {error_msg}"
            
            # Handle text-based action parsing with logging
            if not function_call_found:
                action = await self._convert_text_action_to_function_call(text_content)
                if action:
                    action_response = await self.execute_action(action)
                    return f"{text_content}\n\nParsed and executed action: {action_response}"
            
            return text_content
            
        except Exception as e:
            return f"Error handling response: {str(e)}"
    
    def _extract_response_text(self, response) -> str:
        """Safely extract text content from LLM response."""
        try:
            # Check for safety ratings first
            candidates = getattr(response, 'candidates', [])
            if not candidates:
                return "" #"No text content found in response"
            
            candidate = candidates[0]
            safety_ratings = getattr(candidate, 'safety_ratings', [])
            if safety_ratings:
                for rating in safety_ratings:
                    if hasattr(rating, 'blocked') and rating.blocked:
                        return "Response blocked due to safety rating"
            
            # Try to extract text content
            content = getattr(candidate, 'content', None)
            if not content:
                return "" #"No text content found in response"

            '''parts = getattr(content, 'parts', [])
            if not parts:
                return "" #"No text content found in response"
            
            text_parts = []
            for part in parts:
                if hasattr(part, 'text') and part.text:
                    text_parts.append(part.text)
            
            if text_parts:
                return "\n".join(text_parts)
            
            # Fallback: try direct text access
            if hasattr(response, 'text'):
                return response.text
            
            return "" #"No text content found in response" '''

            parts = getattr(content, 'parts', [])
            text_parts = str(parts)
            return text_parts
            
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
            # Available MCP Tools and their schemas:
            # {self.TOOL_SCHEMAS}
            # Create a prompt specifically for function call generation
            function_call_prompt = f"""You are a function call generator for Unity MCP tools. Your task is to convert a text action into a proper function call.

Text Action: {action_text}
Available tools: {self.tools_str + self.user_defined_tools}
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
                tools=self.available_tools,
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
    
    def _extract_thought_and_action(self, response_text: str) -> tuple[str, str]:
        """Extract thought and action from LLM response text, handling both text and function call formats."""
        thought = ""
        action = ""
        
        if not response_text:
            return thought, action
            
        # Handle structured function call format
        if "function_call" in response_text:
            try:
                # Extract thought if present
                thought_match = re.search(r'Thought:\s*(.+?)(?=\[text:|$)', response_text, re.DOTALL)
                if thought_match:
                    thought = thought_match.group(1).strip()
                
                # Extract function call details
                action = response_text  # Keep the entire function call structure as action
                return thought, action
            except Exception as e:
                print(f"Error parsing function call format: {e}")
                return thought, action
        
        # Handle plain text format
        lines = response_text.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('Thought:'):
                # Save previous section if exists
                if current_section == 'action' and current_content:
                    action = ' '.join(current_content).strip()
                
                # Start new thought section
                current_section = 'thought'
                current_content = [line.replace('Thought:', '').strip()]
                
            elif line.startswith('Action:'):
                # Save previous section if exists
                if current_section == 'thought' and current_content:
                    thought = ' '.join(current_content).strip()
                
                # Start new action section
                current_section = 'action'
                current_content = [line.replace('Action:', '').strip()]
                
            elif line and current_section:
                # Continue current section
                current_content.append(line)
        
        # Save final section
        if current_section == 'thought' and current_content:
            thought = ' '.join(current_content).strip()
        elif current_section == 'action' and current_content:
            action = ' '.join(current_content).strip()
        
        return thought, action
    
    async def run_react_loop(self, goal: str, attachments: List[Dict] = None) -> str:
        """
        Run the ReAct loop with a goal and optional attachments.
        
        Args:
            goal: The goal to achieve
            attachments: List of dicts with 'data' (base64 string), 'mime_type', and 'description'
        """
        self.goal = goal
        self.attachments = attachments or []
        self.current_step = 0
        self.memory = []
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        print(f"🎯 Starting ReAct agent with goal: {goal}")
        print("=" * 80)
        
        while self.current_step < self.max_steps:
            self.current_step += 1
            print(f"\n🔄 STEP {self.current_step}/{self.max_steps}")
            print("=" * 50)
            # Initialize with empty observation - agent will use observe_scene tool when needed
            observation = "No initial observation - agent will use observe_scene tool when needed"
            # REASON & ACT
            print("\n🧠 REASONING AND PLANNING ACTION...")
            print(f"Current Observation: {observation}")
            prompt = self.create_react_prompt(observation)
            
            # Log the prompt being sent to the LLM
            print(f"\n📝 Sending prompt to LLM:")
            print("-" * 40)
            print("-" * 40)
            
            try:
                # Prepare the content parts
                content_parts = []
                
                # Add the text prompt as first part
                content_parts.append(prompt)
                
                # Add image attachments if any
                if self.attachments:
                    for attachment in self.attachments:
                        if attachment.get('mime_type', '').startswith('image/'):
                            content_parts.append({
                                'mime_type': attachment['mime_type'],
                                'data': attachment['data']
                            })
                
                # Generate content with proper parts and tools
                response = self.reasoning_model.generate_content(
                    contents=content_parts,
                    tools=self.available_tools,
                    generation_config=GenerationConfig(
                        temperature=0.4,
                        max_output_tokens=2000,
                        top_p=0.95,
                        top_k=40,
                        stop_sequences=["TASK_COMPLETE", "EXIT", "SUCCESS", "FINISHED"]
                    )
                )
                
                # Extract thought and action from the original LLM response BEFORE processing
                response_text = self._extract_response_text(response)
                thought, action = self._extract_thought_and_action(response_text)
                
                print(f"\n💭 EXTRACTED REASONING:")
                print(f"   Thought: {thought[:200]}{'...' if len(thought) > 200 else ''}")
                print(f"   Action: {action[:200]}{'...' if len(action) > 200 else ''}")
                
                # HANDLE RESPONSE using the _handle_response method
                print("\n🚀 HANDLING LLM RESPONSE...")
                result = await self._handle_response(response)
                print(f"\n📊 ACTION RESULT: {result}")
                
                # Check if task is complete
                if "TASK_COMPLETE" in result:
                    print("\n✅ TASK COMPLETE!")
                    print("=" * 80)
                    return f"Goal '{goal}' successfully completed in {self.current_step} steps!"

                # REFLECT
                print("\n🤔 REFLECTING ON RESULT...")
                reflection = await self.reflect_on_result(action, result)
                print(f"💡 Reflection: {reflection}")

                # STORE IN MEMORY
                memory_entry = {
                    "step": self.current_step,
                    "thought": thought,
                    "action": action,
                    "result": result,
                    "reflection": reflection,
                    "timestamp": datetime.now().isoformat()
                }
                self.memory.append(memory_entry)
                
                print(f"\n💾 MEMORY UPDATED - Step {self.current_step}:")
                print(f"   Thought: {thought[:100]}{'...' if len(thought) > 100 else ''}")
                print(f"   Action: {action[:100]}{'...' if len(action) > 100 else ''}")
                print(f"   Memory entries: {len(self.memory)}")
                
                # Display recent memory entries for context
                if len(self.memory) > 1:
                    print(f"\n📚 RECENT MEMORY CONTEXT:")
                    for i, entry in enumerate(self.memory[-3:], 1):
                        print(f"   {len(self.memory)-3+i}. Step {entry['step']}: {entry['thought'][:50]}{'...' if len(entry['thought']) > 50 else ''}")

                # CHECK GOAL COMPLETION - only if we have a recent observation
                print("\n🎯 CHECKING GOAL COMPLETION...")
                goal_achieved = await self.check_goal_completion(result)

                if goal_achieved:
                    print("\n✅ GOAL ACHIEVED!")
                    print("=" * 80)
                    return f"Goal '{goal}' successfully completed in {self.current_step} steps!"
                
                # Update observation for next iteration
                if "Scene observation completed:" in result:
                    observation_lines = result.split("Scene observation completed:")
                    if len(observation_lines) > 1:
                        observation = observation_lines[1].strip()
                
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
    # 1. FIRST: Render the current scene to 'current_scene.png' using Unity's SceneRenderer
    # 4. RENDER: Render the modified scene to 'modified_scene.png' using Unity's SceneRenderer
    # 5. COMPARE: Use Gemini Vision to compare the images and verify the changes

    # - Always render scenes before and after modifications for visual verification

    SYSTEM_PROMPT = """System prompt: You are an expert in using the Unity game engine with visual feedback capabilities. Your overall task is to answer the query or task specified by the user within the Unity game-engine. You will use a step-by-step approach with visual verification:

1. REASON: Analyze the user's request and plan the necessary observations or modifications
2. EXECUTE: Use MCP tools to modify the Unity scene
3. EVALUATE: Determine if the desired effect has been achieved, by planning to observe the scene visually or its properties through the Unity editor
4. EXIT: If successful, output 'TASK_COMPLETE', otherwise continue with alternative approaches

Follow these guidelines:
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
        self.original_scene_path = "/Users/nmital/Unity projects/MCP_builder3/current_scene.png"
        self.modified_scene_path = "/Users/nmital/Unity projects/MCP_builder3/modified_scene.png"
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
                if (create_response and create_response.content):
                    print("SceneRenderer GameObject created")
                    # Add SceneRenderer component
                    component_response = await self.session.call_tool("manage_gameobject", {
                        "action": "add_component",
                        "objectName": "SceneRenderer",
                        "componentType": "SceneRenderer"
                    })
                    if (component_response and component_response.content):
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

            await asyncio.sleep(10)  # Give time for Play mode and rendering

            # Step 3: Exit Play mode after rendering using manage_editor
            print("Exiting Play mode (manage_editor)...")
            stop_mode_response = await self.session.call_tool("manage_editor", {
                "action": "stop"
            })
            # Wait for Play mode to fully exit
            await asyncio.sleep(5)  # Give time for Play mode to exit
            # Step 4: Find the most recent PNG file and rename it to the target filename
            import os
            import shutil

            # Look for PNG files in the MCP_builder directory
            directory = "/Users/nmital/Unity projects/MCP_builder3"
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

    async def run_react_agent(self, goal: str, attachments: List[Dict]) -> str:
        """Run the ReAct agent with the given goal."""
        react_agent = ReActAgent(self)
        return await react_agent.run_react_loop(goal, attachments)


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


'''
Identify the object tag btr70 and all its components. I have already set the main camera to render as an infrared image by rendering it as a single red channel image.
Modify the materials of the object components so that it resembles the infrared image of an actual btr70 vehicle, where the rear portion where the exhaust is located  appears bright/hot in the infrared image.
'''

'''
Given that I have already set the main camera to render as an infrared image by rendering it as a grayscale image (where only the value of the red channel is rendered).
Modify the materials of the object tagged btr70 and all its children so that it resembles the infrared image of an actual btr70 vehicle, where the rear portion where the exhaust is located  appears bright/hot in the infrared image.
'''

'''
Generate multiple different images of the existing T72 object in the scene, where its individual children parts are ablated.
 For each individual part, also change progressively their emission intensity in the red channel, so that they appear of different brightnesses when rendered. 
'''

'''
The images rendered have already been setup to simulate infrared-like renders. I want you to make the textures (using the emission properties of the objects) match the attached reference image. 
attach: /Users/nmital/Unity projects/MCP_builder3/frame_391.png
reference image to match the rendered image to
'''


'''
I have already setup the camera so that it renders and saves the image as a grayscale from the red channel intensities. The materials are emissive so that it simulates an infrared environment, where high intensities in the red channel lead to bright white pixels in the rendered image, and lower intensities correspond to darker greyish pixel values. Match the emission intensities of the parts of the T72 in the scene to match that of the attached reference image.
'''
'''
'[text: "Action: observe_scene(step=1)\\n"
, function_call {
  name: "observe_scene"
  args {
    fields {
      key: "step"
      value {
        number_value: 1
      }
    }
  }
}
]'
'''