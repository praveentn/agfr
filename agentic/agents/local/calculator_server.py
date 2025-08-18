# agentic/agents/local/calculator_server.py
import math
import logging
from typing import Union, Dict, Any, List
from fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create MCP server
mcp = FastMCP("Calculator Server")

@mcp.tool()
def add(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """Add two numbers together."""
    try:
        result = a + b
        logger.info(f"Addition: {a} + {b} = {result}")
        return int(result) if isinstance(result, float) and result.is_integer() else round(result, 3)
    except Exception as e:
        logger.error(f"Addition failed: {e}")
        raise ValueError(f"Addition failed: {str(e)}")

@mcp.tool()
def subtract(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """Subtract second number from first number."""
    try:
        result = a - b
        logger.info(f"Subtraction: {a} - {b} = {result}")
        return int(result) if isinstance(result, float) and result.is_integer() else round(result, 3)
    except Exception as e:
        logger.error(f"Subtraction failed: {e}")
        raise ValueError(f"Subtraction failed: {str(e)}")

@mcp.tool()
def multiply(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """Multiply two numbers."""
    try:
        result = a * b
        logger.info(f"Multiplication: {a} * {b} = {result}")
        return int(result) if isinstance(result, float) and result.is_integer() else round(result, 3)
    except Exception as e:
        logger.error(f"Multiplication failed: {e}")
        raise ValueError(f"Multiplication failed: {str(e)}")

@mcp.tool()
def divide(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """Divide first number by second number."""
    try:
        if b == 0:
            raise ValueError("Cannot divide by zero")
        result = a / b
        logger.info(f"Division: {a} / {b} = {result}")
        return int(result) if isinstance(result, float) and result.is_integer() else round(result, 3)
    except Exception as e:
        logger.error(f"Division failed: {e}")
        raise ValueError(f"Division failed: {str(e)}")

@mcp.tool()
def power(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """Raise first number to the power of second number."""
    try:
        result = a ** b
        logger.info(f"Power: {a} ^ {b} = {result}")
        return int(result) if isinstance(result, float) and result.is_integer() else round(result, 3)
    except Exception as e:
        logger.error(f"Power calculation failed: {e}")
        raise ValueError(f"Power calculation failed: {str(e)}")

@mcp.tool()
def sqrt(a: Union[int, float]) -> float:
    """Calculate square root of a number."""
    try:
        if a < 0:
            raise ValueError("Cannot calculate square root of negative number")
        result = math.sqrt(a)
        logger.info(f"Square root: √{a} = {result}")
        return round(result, 3)
    except Exception as e:
        logger.error(f"Square root calculation failed: {e}")
        raise ValueError(f"Square root calculation failed: {str(e)}")

@mcp.tool()
def percentage(value: Union[int, float], percent: Union[int, float]) -> float:
    """Calculate percentage of a value."""
    try:
        result = (value * percent) / 100
        logger.info(f"Percentage: {percent}% of {value} = {result}")
        return round(result, 2)
    except Exception as e:
        logger.error(f"Percentage calculation failed: {e}")
        raise ValueError(f"Percentage calculation failed: {str(e)}")

@mcp.tool()
def compound_interest(
    principal: Union[int, float], 
    rate: Union[int, float], 
    time: Union[int, float], 
    frequency: int = 1
) -> Dict[str, Any]:
    """
    Calculate compound interest.
    
    Args:
        principal: Initial amount
        rate: Annual interest rate (as percentage)
        time: Time period in years
        frequency: Compounding frequency per year (default: 1)
    """
    try:
        # Convert percentage to decimal
        rate_decimal = rate / 100
        
        # Compound interest formula: A = P(1 + r/n)^(nt)
        amount = principal * (1 + rate_decimal / frequency) ** (frequency * time)
        interest = amount - principal
        
        result = {
            "principal": round(principal, 2),
            "rate_percent": rate,
            "time_years": time,
            "compound_frequency": frequency,
            "final_amount": round(amount, 2),
            "interest_earned": round(interest, 2),
            "total_return_percent": round((interest / principal) * 100, 2)
        }
        
        logger.info(f"Compound interest calculated: {result}")
        return result
    except Exception as e:
        logger.error(f"Compound interest calculation failed: {e}")
        raise ValueError(f"Compound interest calculation failed: {str(e)}")

@mcp.tool()
def factorial(n: int) -> int:
    """Calculate factorial of a number."""
    try:
        if n < 0:
            raise ValueError("Factorial is not defined for negative numbers")
        if n > 170:  # Prevent overflow
            raise ValueError("Number too large for factorial calculation")
        result = math.factorial(n)
        logger.info(f"Factorial: {n}! = {result}")
        return result
    except Exception as e:
        logger.error(f"Factorial calculation failed: {e}")
        raise ValueError(f"Factorial calculation failed: {str(e)}")

@mcp.tool()
def statistics(numbers: List[Union[int, float]]) -> Dict[str, Any]:
    """Calculate basic statistics for a list of numbers."""
    try:
        if not numbers:
            raise ValueError("Cannot calculate statistics for empty list")
        
        # Convert to floats and validate
        nums = [float(n) for n in numbers]
        
        n = len(nums)
        total = sum(nums)
        mean = total / n
        
        # Calculate variance and standard deviation
        variance = sum((x - mean) ** 2 for x in nums) / n
        std_dev = math.sqrt(variance)
        
        # Sort for median
        sorted_nums = sorted(nums)
        
        # Median
        if n % 2 == 0:
            median = (sorted_nums[n//2 - 1] + sorted_nums[n//2]) / 2
        else:
            median = sorted_nums[n//2]
        
        result = {
            "count": n,
            "sum": round(total, 3),
            "mean": round(mean, 3),
            "median": round(median, 3),
            "min": min(nums),
            "max": max(nums),
            "range": round(max(nums) - min(nums), 3),
            "variance": round(variance, 3),
            "standard_deviation": round(std_dev, 3)
        }
        
        logger.info(f"Statistics calculated for {n} numbers")
        return result
    except Exception as e:
        logger.error(f"Statistics calculation failed: {e}")
        raise ValueError(f"Statistics calculation failed: {str(e)}")

@mcp.tool()
def health() -> Dict[str, Any]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "server": "Calculator Server",
        "timestamp": __import__('time').time(),
        "available_operations": 9
    }

@mcp.tool()
def get_tools() -> List[Dict[str, Any]]:
    """Get list of available tools"""
    return [
        {
            "name": "add",
            "description": "Add two numbers",
            "parameters": {"a": "number", "b": "number"}
        },
        {
            "name": "subtract", 
            "description": "Subtract two numbers",
            "parameters": {"a": "number", "b": "number"}
        },
        {
            "name": "multiply",
            "description": "Multiply two numbers", 
            "parameters": {"a": "number", "b": "number"}
        },
        {
            "name": "divide",
            "description": "Divide two numbers",
            "parameters": {"a": "number", "b": "number"}
        },
        {
            "name": "power",
            "description": "Raise number to power",
            "parameters": {"a": "number (base)", "b": "number (exponent)"}
        },
        {
            "name": "sqrt",
            "description": "Calculate square root",
            "parameters": {"a": "number (non-negative)"}
        },
        {
            "name": "percentage",
            "description": "Calculate percentage of value",
            "parameters": {"value": "number", "percent": "number"}
        },
        {
            "name": "compound_interest",
            "description": "Calculate compound interest",
            "parameters": {
                "principal": "number", 
                "rate": "number (percentage)", 
                "time": "number (years)",
                "frequency": "integer (optional, default: 1)"
            }
        },
        {
            "name": "factorial",
            "description": "Calculate factorial",
            "parameters": {"n": "integer (0-170)"}
        },
        {
            "name": "statistics",
            "description": "Calculate basic statistics",
            "parameters": {"numbers": "array of numbers"}
        }
    ]

if __name__ == "__main__":
    print("=" * 60)
    print("🧮 Starting Calculator MCP Server")
    print("=" * 60)
    print(f"🌐 Port: 9104")
    print(f"⚡ Server Name: Calculator Server")
    print(f"🔧 Available Operations: 10")
    print("=" * 60)
    
    # Run the server
    mcp.run(transport="http", host="0.0.0.0", port=9104)