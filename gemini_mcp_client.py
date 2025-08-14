import asyncio
import sys
import os
from mcp_core import MCPCore
from gui_client import MCPGui
from web_server import app
import uvicorn

def main():
    if len(sys.argv) < 2:
        print("Usage: python gemini_mcp_client.py <path_to_server_script> [--gui|--web]")
        sys.exit(1)

    server_script = sys.argv[1]
    mode = next((arg for arg in sys.argv if arg in ['--gui', '--web']), '')

    if mode == '--web':
        os.environ['UNITY_SERVER_SCRIPT'] = os.path.abspath(server_script)
        uvicorn.run(app, host="0.0.0.0", port=8000)
    elif mode == '--gui':
        gui = MCPGui(server_script)
        gui.run()
    else:
        core = MCPCore()
        asyncio.run(async_cli_main(server_script, core))

async def async_cli_main(server_script: str, core: MCPCore):
    try:
        connected = await core.connect(server_script)
        if not connected:
            print("Failed to connect to MCP server")
            return

        print("MCP Client Started! Type 'quit' to exit.")
        '''while True:
            query = input("\nQuery: ").strip()
            if query.lower() == 'quit':
                break
                
            response = await core.process_request(query)
            print("\nResponse:", response)'''
    finally:
        await core.cleanup()

if __name__ == "__main__":
    main()