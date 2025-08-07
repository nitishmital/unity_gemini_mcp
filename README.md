# Gemini MCP Client

A multi-interface client application for interacting with Unity through Google's Gemini AI and Model Context Protocol (MCP), featuring **visual feedback with scene rendering and comparison**.

## Project Structure

```
GeminiMCP/
├── static/
│   └── css/
│       └── style.css          # Web interface styling
├── templates/
│   └── index.html            # Web interface template
├── mcp_core.py              # Core MCP and Gemini integration with visual feedback
├── gui_client.py            # Tkinter GUI implementation
├── web_server.py            # FastAPI web server
├── unity_mcp_server.py      # Unity MCP server with scene rendering tools
├── SceneRenderer.cs         # Unity C# script for scene rendering
├── test_visual_feedback.py  # Test script for visual feedback system
└── gemini_mcp_client.py     # Main entry point
```

## Features

### 🎯 Visual Feedback System
The system now includes **chain-of-thought reasoning with visual verification**:

1. **Scene Rendering**: Automatically renders Unity scenes before and after modifications
2. **Visual Comparison**: Uses Gemini Vision API to compare scene images
3. **Chain-of-Thought**: Implements step-by-step reasoning with visual evidence
4. **Automatic Verification**: Confirms changes match user requests through visual analysis

### 🔄 Workflow
1. **Render Original**: Enters Play mode → Captures current scene as `original_scene.png` → Exits Play mode
2. **Analyze Request**: Uses chain-of-thought to plan modifications
3. **Execute Changes**: Applies modifications using MCP tools
4. **Render Modified**: Enters Play mode → Captures modified scene as `modified_scene.png` → Exits Play mode
5. **Visual Compare**: Uses Gemini Vision to analyze differences
6. **Verify Success**: Confirms changes match the user's request
7. **Exit or Retry**: Completes task or tries alternative approaches

**Note**: Unity automatically enters and exits Play mode during scene rendering to ensure proper rendering functionality.

## Setup

1. Install dependencies:
```bash
pip install google-generativeai fastapi uvicorn tkinter python-dotenv pillow
```

2. Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your_gemini_api_key_here
```

3. **Unity Setup**:
   - Add `SceneRenderer.cs` to your Unity project
   - Run the setup script to configure the SceneRenderer:
   ```bash
   python setup_unity_scene_renderer.py
   ```
   - This will automatically create the necessary GameObjects and components
   - **Important**: The system automatically handles Play mode entry/exit during scene rendering

## Usage

### Testing Visual Feedback
```bash
python test_visual_feedback.py
```

### Standard Usage
```bash
# Command Line Interface (uses existing Unity MCP server by default)
python gemini_mcp_client.py

# GUI Interface
python gemini_mcp_client.py --gui

# Web Interface
python gemini_mcp_client.py --web

# Or specify a custom server path
python gemini_mcp_client.py /path/to/custom/server.py
```

## Visual Feedback Examples

### Example 1: Object Creation and Movement
```
User: "Create a red cube at position (0, 1, 0) and move it to (2, 1, 0)"

System Response:
1. Renders original scene → original_scene.png
2. Creates cube at (0, 1, 0)
3. Moves cube to (2, 1, 0)
4. Renders modified scene → modified_scene.png
5. Gemini Vision compares images
6. Confirms: "VISUAL_VERIFICATION_SUCCESS - Cube successfully moved"
```

### Example 2: Complex Scene Modification
```
User: "Add a sphere, scale it 2x, and add a light source above"

System Response:
1. Renders original scene
2. Creates sphere
3. Scales sphere to 2x size
4. Adds light source above objects
5. Renders modified scene
6. Gemini Vision analyzes: "Sphere visible and scaled, light positioned correctly"
7. Confirms: "VISUAL_VERIFICATION_SUCCESS"
```

## Core Components

### Enhanced MCP Core (`mcp_core.py`)
- **Visual Reasoning**: Implements chain-of-thought with visual verification
- **Scene Rendering**: Automatic scene capture before/after modifications
- **Image Comparison**: Uses Gemini Vision API for visual analysis
- **Smart Exit**: Exits only when visual verification confirms success

### Enhanced Unity MCP Server (existing server + new tools)
- **read_image_file**: Reads image files and provides base64 data
- **list_image_files**: Lists image files in directories
- **compare_image_files**: Compares two image files for differences
- **manage_gameobject**: Creates and manages GameObjects
- **manage_scene**: Manages Unity scenes
- **manage_script**: Manages C# scripts
- **read_console**: Reads Unity console output

### Scene Renderer (`SceneRenderer.cs`)
- **Flexible Rendering**: Supports custom filenames
- **High Quality**: 1920x1080 resolution by default
- **Easy Integration**: Simple component to add to Unity scenes

## Development

### Adding New Visual Tools
1. **Unity Side**: Add new methods to `SceneRenderer.cs`
2. **MCP Server**: Add corresponding tools to `unity_mcp_server.py`
3. **Core Logic**: Update `mcp_core.py` to handle new visual operations

### Customizing Visual Analysis
- Modify the comparison prompt in `compare_scenes()` method
- Adjust verification criteria in `_handle_response_with_visual_verification()`
- Add new visual analysis types as needed

### Error Handling
- **Rendering Failures**: Falls back to placeholder images
- **Comparison Errors**: Continues with alternative approaches
- **Tool Failures**: Graceful degradation with error reporting

## Advanced Features

### Chain-of-Thought Reasoning
The system implements sophisticated reasoning:
- **Step-by-step analysis** of user requests
- **Visual planning** before execution
- **Iterative refinement** based on visual feedback
- **Alternative approaches** when initial attempts fail

### Visual Verification Levels
- **VISUAL_VERIFICATION_SUCCESS**: Changes match request exactly
- **VISUAL_VERIFICATION_PARTIAL**: Some changes correct, needs refinement
- **VISUAL_VERIFICATION_FAILED**: Changes don't match request

### Performance Optimization
- **Efficient Rendering**: Only renders when modifications are made
- **Smart Caching**: Reuses scene images when appropriate
- **Parallel Processing**: Handles rendering and analysis concurrently

## Contributing

1. Fork the repository
2. Create a feature branch
3. Test visual feedback with your changes
4. Submit a pull request with visual examples

## License

MIT License
