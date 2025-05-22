import asyncio
import sys
from mcp_core import MCPCore
from gui_client import MCPGui

def main():
    if len(sys.argv) < 2:
        print("Usage: python gemini_mcp_client.py <path_to_server_script> [--gui]")
        sys.exit(1)

    use_gui = "--gui" in sys.argv
    server_script = sys.argv[1]
    core = MCPCore()

    if use_gui:
        gui = MCPGui(server_script)
        gui.run()
    else:
        asyncio.run(async_cli_main(server_script, core))

async def async_cli_main(server_script: str, core: MCPCore):
    try:
        connected = await core.connect(server_script)
        if not connected:
            print("Failed to connect to MCP server")
            return

        print("MCP Client Started! Type 'quit' to exit.")
        while True:
            query = input("\nQuery: ").strip()
            if query.lower() == 'quit':
                break
                
            response = await core.process_request(query)
            print("\nResponse:", response)
    finally:
        await core.cleanup()

if __name__ == "__main__":
    main()