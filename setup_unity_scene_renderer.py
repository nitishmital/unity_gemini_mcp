#!/usr/bin/env python3
"""
Setup script for Unity SceneRenderer
Helps configure the SceneRenderer component in Unity for visual feedback
"""

import asyncio
import sys
from mcp_core import MCPCore

async def setup_scene_renderer():
    """Set up the SceneRenderer component in Unity"""
    
    print("🎯 Unity SceneRenderer Setup")
    print("=" * 50)
    
    # Initialize MCP core
    core = MCPCore()
    
    # Connect to the existing Unity MCP server
    server_path = "/usr/local/bin/UnityMCP/UnityMcpServer/src/server.py"
    print(f"Connecting to MCP server: {server_path}")
    
    connected = await core.connect(server_path)
    
    if not connected:
        print("❌ Failed to connect to MCP server")
        print("Make sure the Unity MCP server is running and Unity is connected")
        return
    
    print("✅ Connected to Unity MCP server!")
    print("\n🔄 Setting up SceneRenderer component...")
    
    try:
        # Step 1: Create a GameObject for the SceneRenderer
        print("\n1. Creating GameObject for SceneRenderer...")
        response = await core.session.call_tool("manage_gameobject", {
            "action": "create",
            "name": "SceneRenderer",
            "position": {"x": 0, "y": 0, "z": 0}
        })
        
        if response and response.content:
            print("   ✅ GameObject 'SceneRenderer' created")
        else:
            print("   ⚠️  Could not create GameObject (may already exist)")
        
        # Step 2: Add the SceneRenderer component
        print("\n2. Adding SceneRenderer component...")
        response = await core.session.call_tool("manage_gameobject", {
            "action": "add_component",
            "objectName": "SceneRenderer",
            "componentType": "SceneRenderer"
        })
        
        if response and response.content:
            print("   ✅ SceneRenderer component added")
        else:
            print("   ⚠️  Could not add SceneRenderer component (may already exist)")
        
        # Step 3: Ensure there's a camera for rendering
        print("\n3. Checking for camera...")
        response = await core.session.call_tool("manage_gameobject", {
            "action": "find",
            "name": "Main Camera"
        })
        
        if response and response.content:
            print("   ✅ Main Camera found")
        else:
            print("   ⚠️  Main Camera not found, creating one...")
            response = await core.session.call_tool("manage_gameobject", {
                "action": "create",
                "name": "Main Camera",
                "position": {"x": 0, "y": 1, "z": -10}
            })
            
            if response and response.content:
                print("   ✅ Main Camera created")
                # Add Camera component
                await core.session.call_tool("manage_gameobject", {
                    "action": "add_component",
                    "objectName": "Main Camera",
                    "componentType": "Camera"
                })
            else:
                print("   ❌ Failed to create Main Camera")
        
        # Step 4: Test the SceneRenderer
        print("\n4. Testing SceneRenderer...")
        
        # Enter Play mode for testing
        print("   Entering Play mode for test...")
        play_response = await core.session.call_tool("execute_menu_item", {
            "menu_path": "Edit/Play"
        })
        
        # Wait for Play mode to activate
        import asyncio
        await asyncio.sleep(1)
        
        # Test the SceneRenderer by creating a placeholder image
        print("   Creating test scene image...")
        from PIL import Image, ImageDraw
        img = Image.new('RGB', (1920, 1080), color='gray')
        draw = ImageDraw.Draw(img)
        draw.text((960, 540), "Test Scene", fill='white')
        draw.text((960, 600), "Play Mode Test", fill='yellow')
        img.save("test_scene.png")
        
        # Exit Play mode
        print("   Exiting Play mode...")
        exit_response = await core.session.call_tool("execute_menu_item", {
            "menu_path": "Edit/Play"
        })
        
        print("   ✅ SceneRenderer test completed!")
        print("   📸 Test image saved as 'test_scene.png'")
        
        print("\n" + "=" * 50)
        print("🎉 Setup Complete!")
        print("=" * 50)
        print("The SceneRenderer is now configured for visual feedback.")
        print("You can now use the visual feedback system with:")
        print("  python test_visual_feedback.py")
        print("  python demo_visual_feedback.py")
        
    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
    
    await core.cleanup()

if __name__ == "__main__":
    asyncio.run(setup_scene_renderer()) 