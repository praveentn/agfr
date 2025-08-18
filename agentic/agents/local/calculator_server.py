import ast
import asyncio
import pprint

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

# --- Configuration ---
SERVER_URL = "http://localhost:9104/mcp"  # Calculator server endpoint

pp = pprint.PrettyPrinter(indent=2, width=100)


def unwrap_tool_result(resp):
    """
    Safely unwraps the content from a FastMCP tool call result object.
    """
    if hasattr(resp, "content") and resp.content:
        content_object = resp.content[0]
        if hasattr(content_object, "json"):
            return content_object.json
        if hasattr(content_object, "text"):
            try:
                return ast.literal_eval(content_object.text)
            except (ValueError, SyntaxError):
                return content_object.text
    return resp


async def main():
    transport = StreamableHttpTransport(url=SERVER_URL)
    client = Client(transport)

    print("\n🚀 Connecting to Calculator MCP server at:", SERVER_URL)
    async with client:
        # 1. Ping to test connectivity
        print("\n🔗 Testing server connectivity...")
        await client.ping()
        print("✅ Server is reachable!\n")

        # 2. Discover server capabilities
        print("🛠️  Available tools:")
        tools = await client.list_tools()
        pp.pprint(tools)

        # 3. Test each calculator tool
        print("\n➕ Testing tool: add")
        add_resp = await client.call_tool("add", {"a": 5, "b": 3})
        print("Result:", unwrap_tool_result(add_resp))

        print("\n➖ Testing tool: subtract")
        sub_resp = await client.call_tool("subtract", {"a": 10, "b": 4})
        print("Result:", unwrap_tool_result(sub_resp))

        print("\n✖️ Testing tool: multiply")
        mul_resp = await client.call_tool("multiply", {"a": 6, "b": 7})
        print("Result:", unwrap_tool_result(mul_resp))

        print("\n➗ Testing tool: divide")
        div_resp = await client.call_tool("divide", {"a": 20, "b": 5})
        print("Result:", unwrap_tool_result(div_resp))

        print("\n⚡ Testing tool: power")
        pow_resp = await client.call_tool("power", {"a": 2, "b": 8})
        print("Result:", unwrap_tool_result(pow_resp))


if __name__ == "__main__":
    asyncio.run(main())
