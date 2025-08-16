# agentic/agents/local/calculator_server.py
from fastmcp import FastMCP
from typing import Union

mcp = FastMCP("Calculator Server")

@mcp.tool()
def add(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """Add two numbers together"""
    return a + b

@mcp.tool()
def multiply(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """Multiply two numbers"""
    return a * b

@mcp.tool()
def divide(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """Divide two numbers. Returns error if dividing by zero."""
    if b == 0:
        raise ValueError("Cannot divide by zero!")
    return a / b

@mcp.tool()
def subtract(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """Subtract second number from first number"""
    return a - b

@mcp.tool()
def power(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """Raise first number to the power of second number"""
    return a ** b

if __name__ == "__main__":
    print("Starting Calculator Server on port 9104...")
    mcp.run(transport="sse", host="0.0.0.0", port=9104)

