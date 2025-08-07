#!/usr/bin/env python3
"""
Test script for the ReAct agent implementation.
This demonstrates the ReAct-style reasoning loop for Unity scene manipulation.
"""

import asyncio
import sys
import os

# Add the current directory to the path so we can import mcp_core
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mcp_core import MCPCore

async def test_react_agent():
    """Test the ReAct agent with a simple Unity scene manipulation goal."""
    
    # Initialize MCP Core
    mcp_core = MCPCore()
    
    # Connect to Unity MCP server
    server_path = "./unity_mcp_server.py"  # Adjust path as needed
    
    print("🔌 Connecting to Unity MCP server...")
    connected = await mcp_core.connect(server_path)
    
    if not connected:
        print("❌ Failed to connect to Unity MCP server")
        return
    
    print("✅ Connected to Unity MCP server")
    
    # Test goal
    goal = "Create a red cube and place it on top of a green platform"
    
    print(f"\n🎯 Testing ReAct agent with goal: {goal}")
    print("=" * 60)
    
    try:
        # Run the ReAct agent
        result = await mcp_core.run_react_agent(goal)
        print(f"\n📋 Final Result: {result}")
        
    except Exception as e:
        print(f"❌ Error during ReAct agent execution: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        await mcp_core.cleanup()
        print("\n🧹 Cleanup completed")

async def test_simple_goal():
    """Test with a simpler goal for debugging."""
    
    mcp_core = MCPCore()
    server_path = "./unity_mcp_server.py"
    
    print("🔌 Connecting to Unity MCP server...")
    connected = await mcp_core.connect(server_path)
    
    if not connected:
        print("❌ Failed to connect to Unity MCP server")
        return
    
    print("✅ Connected to Unity MCP server")
    
    # Simple goal for testing
    goal = "Render the current scene to see what's there"
    
    print(f"\n🎯 Testing simple goal: {goal}")
    print("=" * 60)
    
    try:
        result = await mcp_core.run_react_agent(goal)
        print(f"\n📋 Final Result: {result}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await mcp_core.cleanup()

if __name__ == "__main__":
    print("🚀 Starting ReAct Agent Test")
    print("=" * 60)
    
    # Check if GOOGLE_API_KEY is set
    if not os.getenv('GOOGLE_API_KEY'):
        print("❌ Error: GOOGLE_API_KEY environment variable not set")
        print("Please set your Google API key before running this test")
        sys.exit(1)
    
    # Run the test
    asyncio.run(test_simple_goal()) 