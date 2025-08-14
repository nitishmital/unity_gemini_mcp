# Gemini MCP Client

A multi-interface client application for interacting with Unity through Google's Gemini AI and Model Context Protocol (MCP).

## Project Structure

```
GeminiMCP/
├── static/
│   └── css/
│       └── style.css          # Web interface styling
├── templates/
│   └── index.html            # Web interface template
├── mcp_core.py              # Core MCP and Gemini integration
├── gui_client.py            # Tkinter GUI implementation
├── web_server.py            # FastAPI web server
└── gemini_mcp_client.py     # Main entry point
```

## Setup

1. Install dependencies:
```bash
pip install google-generativeai fastapi uvicorn tkinter python-dotenv
```

2. Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your_gemini_api_key_here
```

3. Ensure you have a Unity MCP server script ready to use.

## Usage

The application supports three modes of operation:

### 1. Command Line Interface
```bash
python gemini_mcp_client.py path/to/server_script
```

### 2. GUI Interface
```bash
python gemini_mcp_client.py path/to/server_script --gui
```

### 3. Web Interface
```bash
python gemini_mcp_client.py path/to/server_script --web
```
Then open `http://localhost:8000` in your browser.

## Core Components

- **mcp_core.py**: Handles communication with Gemini AI and MCP server
- **gui_client.py**: Provides a desktop GUI using Tkinter
- **web_server.py**: Implements a web server using FastAPI and WebSocket
- **gemini_mcp_client.py**: Main entry point with mode selection

## Development

### Adding New Features

1. Core Logic Changes:
   - Modify `mcp_core.py` for changes to AI or MCP interaction
   - Update the SYSTEM_PROMPT for changes to AI behavior

2. UI Changes:
   - GUI: Modify `gui_client.py` for desktop interface changes
   - Web: Update `templates/index.html` and `static/css/style.css`

3. Adding New Modes:
   - Add mode flag in `gemini_mcp_client.py`
   - Create new interface implementation
   - Update main() to handle new mode

## Error Handling

- Connection errors are handled in `mcp_core.py`
- Each interface implements its own error display
- Check logs for detailed error messages

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License
