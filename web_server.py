from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import asyncio
from mcp_core import MCPCore
import uvicorn
import os

app = FastAPI()

# Ensure static directory exists
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def get():
    with open("templates/index.html") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws/{server_script}")
async def websocket_endpoint(websocket: WebSocket, server_script: str):
    await websocket.accept()
    core = MCPCore()
    
    try:
        connected = await core.connect(server_script)
        if not connected:
            await websocket.send_text("Failed to connect to MCP server")
            return

        await websocket.send_text("Connected to MCP server")
        
        while True:
            query = await websocket.receive_text()
            if query.lower() == 'quit':
                break
                
            response = await core.process_request(query)
            await websocket.send_text(response)
    except Exception as e:
        await websocket.send_text(f"Error: {str(e)}")
    finally:
        await core.cleanup()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
