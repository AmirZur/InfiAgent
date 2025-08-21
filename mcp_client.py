import asyncio
import json
import os
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()  # load environment variables from .env

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        
        endpoint = os.getenv("ENDPOINT_URL")
        subscription_key = os.getenv("AZURE_OPENAI_API_KEY")

        self.deployment = os.getenv("DEPLOYMENT_NAME")
        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=subscription_key,
            api_version="2024-05-01-preview"
        )

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
        self.tools = response.tools
        print(f"Connected! Discovered {len(self.tools)} tools: {[tool.name for tool in self.tools]}")

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]
        
        tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            }
            for tool in self.tools
        ]

        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=messages,
            max_tokens=1000,
            temperature=0.7,
            top_p=0.95,
            tools=tools,
        )

        # Process response and handle tool calls
        final_text = []
        while True:
            content = response.choices[0]
            if content.finish_reason == 'stop':
                final_text.append(content.message.content)
                break
            elif content.finish_reason == 'tool_calls':
                for tool_call in content.message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments) # convert from str to dict
                
                    # Execute tool call
                    result = await self.session.call_tool(tool_name, tool_args)
                    final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")
                    final_text.append(f"[Tool response: {result.content}]")

                    messages.append({
                        "role": "assistant", 
                        "content": f"[Calling tool {tool_name} with args {tool_args}]"
                    })

                    messages.append({
                        "role": "user", 
                        "content": result.content
                    })

                    # get model's response to tool call
                    response = self.client.chat.completions.create(
                        model=self.deployment,
                        messages=messages,
                        max_tokens=1000,
                        temperature=0.7,
                        top_p=0.95,
                        tools=tools
                    )
            else:
                print("Finish reason:", content.finish_reason)

        return "\n".join(final_text)

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