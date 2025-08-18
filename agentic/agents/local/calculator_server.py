# agentic/agents/local/calculator_server.py
import math
from typing import Union, Dict, Any
from fastmcp import FastMCP

mcp = FastMCP("Calculator Server")

@mcp.tool()
def add(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """Add two numbers together."""
    try:
        result = a + b
        return int(result) if result.is_integer() else round(result, 3)
    except Exception as e:
        raise ValueError(f"Addition failed: {str(e)}")

@mcp.tool()
def subtract(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """Subtract second number from first number."""
    try:
        result = a - b
        return int(result) if result.is_integer() else round(result, 3)
    except Exception as e:
        raise ValueError(f"Subtraction failed: {str(e)}")

@mcp.tool()
def multiply(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """Multiply two numbers."""
    try:
        result = a * b
        return int(result) if result.is_integer() else round(result, 3)
    except Exception as e:
        raise ValueError(f"Multiplication failed: {str(e)}")

@mcp.tool()
def divide(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """Divide first number by second number."""
    try:
        if b == 0:
            raise ValueError("Cannot divide by zero")
        result = a / b
        return int(result) if result.is_integer() else round(result, 3)
    except Exception as e:
        raise ValueError(f"Division failed: {str(e)}")

@mcp.tool()
def power(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """Raise first number to the power of second number."""
    try:
        result = a ** b
        return int(result) if result.is_integer() else round(result, 3)
    except Exception as e:
        raise ValueError(f"Power calculation failed: {str(e)}")

@mcp.tool()
def sqrt(a: Union[int, float]) -> float:
    """Calculate square root of a number."""
    try:
        if a < 0:
            raise ValueError("Cannot calculate square root of negative number")
        result = math.sqrt(a)
        return round(result, 3)
    except Exception as e:
        raise ValueError(f"Square root calculation failed: {str(e)}")

@mcp.tool()
def percentage(value: Union[int, float], percent: Union[int, float]) -> float:
    """Calculate percentage of a value."""
    try:
        result = (value * percent) / 100
        return round(result, 2)
    except Exception as e:
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
        
        return {
            "principal": round(principal, 2),
            "rate_percent": rate,
            "time_years": time,
            "compound_frequency": frequency,
            "final_amount": round(amount, 2),
            "interest_earned": round(interest, 2),
            "total_return_percent": round((interest / principal) * 100, 2)
        }
    except Exception as e:
        raise ValueError(f"Compound interest calculation failed: {str(e)}")

@mcp.tool()
def factorial(n: int) -> int:
    """Calculate factorial of a number."""
    try:
        if n < 0:
            raise ValueError("Factorial is not defined for negative numbers")
        if n > 170:  # Prevent overflow
            raise ValueError("Number too large for factorial calculation")
        return math.factorial(n)
    except Exception as e:
        raise ValueError(f"Factorial calculation failed: {str(e)}")

@mcp.tool()
def logarithm(a: Union[int, float], base: Union[int, float] = math.e) -> float:
    """Calculate logarithm of a number with given base (default: natural log)."""
    try:
        if a <= 0:
            raise ValueError("Logarithm is not defined for non-positive numbers")
        if base <= 0 or base == 1:
            raise ValueError("Invalid base for logarithm")
        
        if base == math.e:
            result = math.log(a)
        else:
            result = math.log(a) / math.log(base)
        
        return round(result, 6)
    except Exception as e:
        raise ValueError(f"Logarithm calculation failed: {str(e)}")

@mcp.tool()
def trigonometry(value: Union[int, float], function: str, unit: str = "radians") -> float:
    """
    Calculate trigonometric functions.
    
    Args:
        value: Input value
        function: sin, cos, tan, asin, acos, atan
        unit: radians or degrees
    """
    try:
        # Convert degrees to radians if needed
        if unit.lower() == "degrees":
            if function.startswith("a"):  # inverse functions
                # For inverse functions, input is unitless, output needs conversion
                pass
            else:
                value = math.radians(value)
        
        if function == "sin":
            result = math.sin(value)
        elif function == "cos":
            result = math.cos(value)
        elif function == "tan":
            result = math.tan(value)
        elif function == "asin":
            if not -1 <= value <= 1:
                raise ValueError("Input for asin must be between -1 and 1")
            result = math.asin(value)
            if unit.lower() == "degrees":
                result = math.degrees(result)
        elif function == "acos":
            if not -1 <= value <= 1:
                raise ValueError("Input for acos must be between -1 and 1")
            result = math.acos(value)
            if unit.lower() == "degrees":
                result = math.degrees(result)
        elif function == "atan":
            result = math.atan(value)
            if unit.lower() == "degrees":
                result = math.degrees(result)
        else:
            raise ValueError(f"Unknown trigonometric function: {function}")
        
        return round(result, 6)
    except Exception as e:
        raise ValueError(f"Trigonometry calculation failed: {str(e)}")

@mcp.tool()
def statistics(numbers: list) -> Dict[str, Any]:
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
        
        # Sort for median and quartiles
        sorted_nums = sorted(nums)
        
        # Median
        if n % 2 == 0:
            median = (sorted_nums[n//2 - 1] + sorted_nums[n//2]) / 2
        else:
            median = sorted_nums[n//2]
        
        return {
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
    except Exception as e:
        raise ValueError(f"Statistics calculation failed: {str(e)}")

if __name__ == "__main__":
    print("Starting Calculator Server on port 9104...")
    
    # Use sync run to avoid asyncio conflicts
    mcp.run(transport="http", host="0.0.0.0", port=9104)
