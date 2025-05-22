import asyncio
import google.generativeai as genai
from google.generativeai import types, GenerationConfig
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack
from typing import Optional, List, Dict
import os
from dotenv import load_dotenv
import proto

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

class MCPCore:
    SYSTEM_PROMPT = """System prompt: Your overall task is to answer the query or task specified by the user within the Unity game-engine. To do that you are provided with a list of tools from the Model Context Protocol for Unity Game Engine. After each step, you will be provided with the result of your action. Check if the action taken indeed resulted in the task being accomplished. If yes, output 'EXIT', or 'success', or 'finished', else output 'Continue'. If you continue, try different approaches to find the solution. If you do not find an answer, just exit. Do not try to make up an answer. """
    
    ''' """You are a Unity scene construction assistant. Follow these guidelines:
    1. Analyze the requested changes carefully
    2. Break down complex operations into simple steps
    3. Verify each operation's success before proceeding
    4. Use precise coordinates and rotations
    5. Handle errors gracefully and suggest alternatives
    6. Confirm final results match the user's request
    
    Available tools: {tools}
    
    Current request: """ '''

    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.available_tools = []
        self.chat_history = []

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
        # Process and clean up tool definitions
        self._clean_schema(tools)
        return [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name=tool.name,
                        description=tool.description,
                        parameters=tool.inputSchema #self._clean_schema(tool.inputSchema)
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

    async def process_request(self, query: str) -> str:
        model = genai.GenerativeModel('models/gemini-2.0-flash')
        prompt = self.SYSTEM_PROMPT + "\n Tools from Model Context Protocol: " + self.tools + "\n Query: " + query #self.SYSTEM_PROMPT.format(tools=self.tools) + query
        try:
            response = await self._execute_with_reasoning(model, prompt)
            return response
        except Exception as e:
            return f"Error processing request: {e}"

    async def _execute_with_reasoning(self, model, prompt: str) -> str:
        conversation = []
        max_turns = 5
        initial_prompt = prompt
        for turn in range(max_turns):
            print(f"Turn {turn + 1}/{max_turns}")
            response = model.generate_content(
                contents=prompt,
                tools=self.available_tools,
                generation_config=GenerationConfig(temperature=0.8, max_output_tokens=3000, top_p=0.95, top_k=40, stop_sequences=["TASK_COMPLETE", "EXIT", "SUCCESS", "FINISHED"])
            )
            
            result = await self._handle_response(model, response, initial_prompt=prompt)
            conversation.append(result)
            
            if "TASK_COMPLETE" in result:
                break
                
            prompt += f"\nResult: {result}\nNext step. \n"
            
        return "\n".join(conversation)

    async def _handle_response(self, model, response, initial_prompt) -> str:
        try:
            result_parts = []
            for candidate in response.candidates:
                for part in candidate.content.parts:
                    if part.text:
                        result_parts.append(f"Reasoning: {part.text}")
                    
                    elif part.function_call:
                        # Extract function call details
                        function = proto.Message.to_dict(part.function_call)
                        tool_name = function["name"]
                        tool_args = function["args"]
                        
                        result_parts.append(f"\nExecuting: {tool_name}")
                        
                        # Call the tool and get response
                        try:
                            tool_response = await self.session.call_tool(tool_name, tool_args)
                            if tool_response and tool_response.content:
                                response_text = tool_response.content[0].text
                                result_parts.append(f"Unity MCP Tool response: {response_text}")
                                # Get next response from Gemini
                                result_parts.extend([
                                    f"\nAnalyzing tool response..."
                                ])
                                response_analysis_prompt = ""
                                analysis_prompt = initial_prompt + " ".join(result_parts) + " " + response_analysis_prompt
                                #print("analysis_prompt:", analysis_prompt)
                                response = model.generate_content(
                                    contents=analysis_prompt,
                                    tools=self.available_tools,
                                    generation_config=GenerationConfig(temperature=0.8)
                                )
                                
                                # Check if the operation was successful
                                analysis_response_text = response.candidates[0].content.parts[0].text
                                if "success" in analysis_response_text.lower() or "complete" in analysis_response_text.lower():
                                    result_parts.append("TASK_COMPLETE")
                                elif "error" in response_text.lower() or "failed" in response_text.lower():
                                    result_parts.append("Operation failed, trying alternative approach...")
                            else:
                                result_parts.append("No response from tool")
                        except Exception as e:
                            result_parts.append(f"Tool execution error: {str(e)}")
            
            if not result_parts:
                return "No valid response generated"
                
            return "\n".join(result_parts)
            
        except Exception as e:
            return f"Error handling response: {str(e)}"

    async def cleanup(self):
        await self.exit_stack.aclose()
