#!/usr/bin/env python3
"""
Examples of using the ReAct agent for Unity scene manipulation.
This file demonstrates different types of goals and how the ReAct agent handles them.
"""

import asyncio
import sys
import os
from typing import List, Dict
import base64

# Add the current directory to the path so we can import mcp_core
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mcp_core import MCPCore, ReActAgent

class ReActAgentExamples:
    """Examples of using the ReAct agent for different Unity scene manipulation tasks."""
    
    def __init__(self):
        self.mcp_core = None
        self.react_agent = None
    
    async def setup(self):
        """Initialize the MCP connection and ReAct agent."""
        self.mcp_core = MCPCore()
        server_path = "/usr/local/bin/UnityMCP/UnityMcpServer/src/server.py"
        
        print("🔌 Connecting to Unity MCP server...")
        connected = await self.mcp_core.connect(server_path)
        
        if not connected:
            raise Exception("Failed to connect to Unity MCP server")
        
        print("✅ Connected to Unity MCP server")
        self.react_agent = ReActAgent(self.mcp_core)
    
    async def cleanup(self):
        """Clean up resources."""
        if self.mcp_core:
            await self.mcp_core.cleanup()
    
    async def example_1_simple_scene_observation(self):
        """Example 1: Simple scene observation and description."""
        print("\n" + "="*60)
        print("📋 Example 1: Simple Scene Observation")
        print("="*60)
        
        goal = "Observe the current Unity scene and describe what you see"
        
        result = await self.react_agent.run_react_loop(goal)
        print(f"\n📋 Result: {result}")
    
    async def example_2_object_creation(self):
        """Example 2: Creating objects in the scene."""
        print("\n" + "="*60)
        print("📋 Example 2: Object Creation")
        print("="*60)
        
        goal = "Create a red cube at position (0, 1, 0) and a blue sphere at position (2, 0, 0)"
        
        result = await self.react_agent.run_react_loop(goal)
        print(f"\n📋 Result: {result}")
    
    async def example_3_object_manipulation(self):
        """Example 3: Moving and transforming objects."""
        print("\n" + "="*60)
        print("📋 Example 3: Object Manipulation")
        print("="*60)
        
        goal = "Find the red cube and move it to position (0, 2, 0), then rotate it 45 degrees around the Y axis"
        
        result = await self.react_agent.run_react_loop(goal)
        print(f"\n📋 Result: {result}")
    
    async def example_4_complex_scene_setup(self):
        """Example 4: Complex scene setup with multiple objects."""
        print("\n" + "="*60)
        print("📋 Example 4: Complex Scene Setup")
        print("="*60)
        
        goal = """Create a scene with:
        1. A green platform at position (0, 0, 0) with scale (3, 0.5, 3)
        2. A red cube at position (0, 1, 0)
        3. A blue sphere at position (2, 0.5, 0)
        4. A yellow cylinder at position (-2, 0.5, 0)
        Then move the red cube to sit on top of the green platform"""
        
        result = await self.react_agent.run_react_loop(goal)
        print(f"\n📋 Result: {result}")
    
    async def example_5_physics_simulation(self):
        """Example 5: Setting up physics simulation."""
        print("\n" + "="*60)
        print("📋 Example 5: Physics Simulation")
        print("="*60)
        
        goal = "Create a scene with a ground plane and a ball, then enable physics simulation to see the ball fall"
        
        result = await self.react_agent.run_react_loop(goal)
        print(f"\n📋 Result: {result}")
    
    async def example_6_lighting_setup(self):
        """Example 6: Setting up lighting and materials."""
        print("\n" + "="*60)
        print("📋 Example 6: Lighting Setup")
        print("="*60)
        
        goal = "Create a scene with proper lighting - add a directional light and create objects with different materials"
        
        result = await self.react_agent.run_react_loop(goal)
        print(f"\n📋 Result: {result}")
    
    async def run_all_examples(self):
        """Run all examples in sequence."""
        try:
            await self.setup()
            
            # Run examples
            await self.example_1_simple_scene_observation()
            await self.example_2_object_creation()
            await self.example_3_object_manipulation()
            await self.example_4_complex_scene_setup()
            await self.example_5_physics_simulation()
            await self.example_6_lighting_setup()
            
        except Exception as e:
            print(f"❌ Error running examples: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            await self.cleanup()
    
    async def run_specific_example(self, example_number: int, attachments: List[Dict] = None):
        """Run a specific example by number with optional attachments."""
        try:
            await self.setup()
            
            examples = {
                1: self.example_1_simple_scene_observation,
                2: self.example_2_object_creation,
                3: self.example_3_object_manipulation,
                4: self.example_4_complex_scene_setup,
                5: self.example_5_physics_simulation,
                6: self.example_6_lighting_setup
            }
            
            if example_number in examples:
                # Create example-specific attachments if needed
                example_attachments = attachments or []
                result = await self.mcp_core.run_react_agent(examples[example_number].__doc__, example_attachments)
                print(f"\n📋 Result: {result}")
            else:
                print(f"❌ Example {example_number} not found. Available examples: {list(examples.keys())}")
        
        except Exception as e:
            print(f"❌ Error running example {example_number}: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            await self.cleanup()

async def interactive_mode():
    """Run the ReAct agent in interactive mode where users can input their own goals."""
    print("\n🎮 Interactive ReAct Agent Mode")
    print("="*60)
    print("Enter your Unity scene manipulation goals and watch the ReAct agent work!")
    print("Type 'quit' to exit, 'help' for example goals")
    print("="*60)
    
    examples = [
        "Create a red cube and place it on a green platform",
        "Build a simple house with walls and a roof",
        "Create a scene with multiple colored spheres arranged in a circle",
        "Set up a physics simulation with falling objects",
        "Create a scene with proper lighting and materials"
    ]
    
    mcp_core = MCPCore()
    server_path = "/usr/local/bin/UnityMCP/UnityMcpServer/src/server.py"
    
    print("🔌 Connecting to Unity MCP server...")
    connected = await mcp_core.connect(server_path)
    
    if not connected:
        print("❌ Failed to connect to Unity MCP server")
        return
    
    print("✅ Connected to Unity MCP server")
    
    try:
        overall_goal = ""
        attachments = []
        while True:
            print("\n🎯 Enter your goal (or 'help' for examples, 'attach' to add attachments, 'quit' to exit):")
            goal = input("> ").strip()
            
            if goal.lower() == 'quit':
                print("Quitting!")
                return
            elif goal.lower() == 'done':
                break
            elif goal.lower() == 'help':
                print("\n📚 Example goals:")
                for i, example in enumerate(examples, 1):
                    print(f"{i}. {example}")
                continue
            elif goal.lower() == 'attach':

                while True:
                    print("\nEnter attachment path (or 'done' to finish):")
                    path = input("> ").strip()
                    if path.lower() == 'done':
                        break
                    
                    if os.path.exists(path):
                        print("Enter attachment description:")
                        description = input("> ").strip()
                        with open(path, "rb") as f:
                            data = base64.b64encode(f.read()).decode()
                        mime_type = "image/png" if path.endswith('.png') else "application/octet-stream"
                        attachments.append({
                            "data": data,
                            "mime_type": mime_type,
                            "description": description
                        })
                        print("✅ Attachment added")
                    else:
                        print("❌ File not found")
                continue
            elif not goal:
                continue
            else:
                overall_goal += goal
            
        print(f"\n🚀 Running ReAct agent with goal: {goal} and attachments: {attachments}")
        print("="*60)
            
        try:
            result = await mcp_core.run_react_agent(overall_goal, attachments)
            print(f"\n📋 Result: {result}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    finally:
        await mcp_core.cleanup()

if __name__ == "__main__":
    print("🚀 ReAct Agent Examples")
    print("="*60)
    
    # Check if GOOGLE_API_KEY is set
    if not os.getenv('GOOGLE_API_KEY'):
        print("❌ Error: GOOGLE_API_KEY environment variable not set")
        print("Please set your Google API key before running this script")
        sys.exit(1)

    asyncio.run(interactive_mode()) # if running in debug mode
    # Parse command line arguments
    '''if len(sys.argv) > 1:
        if sys.argv[1] == "interactive":
            asyncio.run(interactive_mode())
        elif sys.argv[1].isdigit():
            example_num = int(sys.argv[1])
            examples = ReActAgentExamples()
            asyncio.run(examples.run_specific_example(example_num))
        else:
            print("Usage:")
            print("  python react_agent_examples.py                    # Run all examples")
            print("  python react_agent_examples.py <number>          # Run specific example")
            print("  python react_agent_examples.py interactive       # Interactive mode")
    else:
        # Run all examples
        examples = ReActAgentExamples()
        asyncio.run(examples.run_all_examples())'''