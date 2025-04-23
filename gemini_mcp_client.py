

import asyncio

from google.genai.types import GenerateContentConfig
from google.protobuf.json_format import MessageToJson, MessageToDict
from mcp import ClientSession, StdioServerParameters
from dotenv import load_dotenv
import os
from google import genai as generativeai
import google.generativeai as genai
from google.generativeai import types
from mcp.client.stdio import stdio_client
from typing import Optional
from contextlib import AsyncExitStack

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

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process a query using Gemini and available tools"""

        system_prompt = '''
        System prompt: Your overall task is to accomplish the task specified in the query to do within the Unity game-engine. 
To do that you are provided with a list of tools from the Model Context Protocol. Reason through the steps you would need to go through to fulfil the task. 
After each step, you will be provided with the result of your action. Check if the action taken indeed resulted in the task being accomplished. If yes, output 'EXIT', else 
 output 'Continue'. \n
Query: '''
        prompt = query
        response = await self.session.list_tools()
        for tool in response.tools:
            remove_key(tool.inputSchema, "title")
            remove_key(tool.inputSchema, "default")
            remove_key(tool.inputSchema, "additionalProperties")

        available_tools = [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name=tool.name,
                        description=tool.description,
                        parameters=tool.inputSchema
                    )
                ]
            )
            for tool in response.tools
        ]
        # Gemini API call
        assistant_message_content = system_prompt + prompt + "\n"
        model = genai.GenerativeModel('models/gemini-2.0-flash')
        for i in range(3):
            response = model.generate_content(
                contents=assistant_message_content,
                tools=available_tools,
            )
            print(response)
            # Process response and handle tool calls
            for candidate in response.candidates:
                for part in candidate.content.parts:
                    if part.text:
                        assistant_message_content = assistant_message_content + "AI response: " + part.text + "\n"
                    elif part.function_call:
                        function = proto.Message.to_dict(part.function_call)
                        print("function: ", function)
                        tool_name = function["name"]
                        tool_args = function["args"]  # This is a dictionary
                        print(f"Calling function: {tool_name} with args: {tool_args}")
                        # Execute tool call
                        result = await self.session.call_tool(tool_name, tool_args)
                        print("result: ", result)
                        assistant_message_content = assistant_message_content + "AI response: " + result.content[0].text + "\n"
                        # Get next response from Gemini
                        response = model.generate_content(
                            contents=assistant_message_content,
                            tools=available_tools
                            )
                        print('second response: ', response.candidates[0].content)
                        assistant_message_content = assistant_message_content + "AI response: " + response.candidates[0].content.parts[0].text + "\n"
                        if 'EXIT' in response.candidates[0].content.parts[0].text:
                            print("***************** Success. Exiting. **************************")
                            return assistant_message_content

        return assistant_message_content

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())


##############################################################################################

