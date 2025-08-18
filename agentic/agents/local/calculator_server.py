# agentic/agents/local/calculator_server.py
import json
import time
import uuid
import math
import logging
import asyncio
from typing import Union, Dict, Any, List, Optional
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MCP Server Implementation
class CalculatorMCPServer:
    def __init__(self):
        self.app = FastAPI(title="Calculator MCP Server")
        self.sessions = {}
        self.tools = [
            {
                "name": "add",
                "description": "Add two numbers together",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "First number"},
                        "b": {"type": "number", "description": "Second number"}
                    },
                    "required": ["a", "b"]
                }
            },
            {
                "name": "subtract",
                "description": "Subtract second number from first number",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "First number"},
                        "b": {"type": "number", "description": "Second number"}
                    },
                    "required": ["a", "b"]
                }
            },
            {
                "name": "multiply",
                "description": "Multiply two numbers",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "First number"},
                        "b": {"type": "number", "description": "Second number"}
                    },
                    "required": ["a", "b"]
                }
            },
            {
                "name": "divide",
                "description": "Divide first number by second number",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "Dividend"},
                        "b": {"type": "number", "description": "Divisor"}
                    },
                    "required": ["a", "b"]
                }
            },
            {
                "name": "power",
                "description": "Raise first number to the power of second number",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "Base number"},
                        "b": {"type": "number", "description": "Exponent"}
                    },
                    "required": ["a", "b"]
                }
            },
            {
                "name": "sqrt",
                "description": "Calculate square root of a number",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "Number to find square root of", "minimum": 0}
                    },
                    "required": ["a"]
                }
            },
            {
                "name": "percentage",
                "description": "Calculate percentage of a value",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "number", "description": "Base value"},
                        "percent": {"type": "number", "description": "Percentage to calculate"}
                    },
                    "required": ["value", "percent"]
                }
            },
            {
                "name": "compound_interest",
                "description": "Calculate compound interest for investments",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "principal": {"type": "number", "description": "Initial investment amount", "minimum": 0},
                        "rate": {"type": "number", "description": "Annual interest rate (as percentage)", "minimum": 0},
                        "time": {"type": "number", "description": "Time period in years", "minimum": 0},
                        "frequency": {"type": "integer", "description": "Compounding frequency per year", "default": 1, "minimum": 1}
                    },
                    "required": ["principal", "rate", "time"]
                }
            },
            {
                "name": "statistics",
                "description": "Calculate basic statistics for a list of numbers",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "numbers": {"type": "array", "items": {"type": "number"}, "description": "List of numbers to analyze", "minItems": 1}
                    },
                    "required": ["numbers"]
                }
            },
            {
                "name": "health",
                "description": "Health check for the server",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                }
            }
        ]
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self.setup_routes()
    
    def setup_routes(self):
        @self.app.post("/")
        async def mcp_endpoint_post(
            request: Request,
            mcp_protocol_version: Optional[str] = Header(None, alias="MCP-Protocol-Version"),
            mcp_session_id: Optional[str] = Header(None, alias="Mcp-Session-Id"),
            origin: Optional[str] = Header(None)
        ):
            # Security: Validate Origin header
            if origin and not self._is_allowed_origin(origin):
                raise HTTPException(status_code=403, detail="Origin not allowed")
            
            try:
                body = await request.body()
                message = json.loads(body.decode('utf-8'))
                
                if message.get("method") == "initialize":
                    return await self._handle_initialize(message, mcp_protocol_version)
                elif message.get("method") == "initialized":
                    return JSONResponse(status_code=202, content={})
                elif message.get("method") == "tools/list":
                    return await self._handle_list_tools(message, mcp_session_id)
                elif message.get("method") == "tools/call":
                    return await self._handle_call_tool(message, mcp_session_id)
                else:
                    error_response = {
                        "jsonrpc": "2.0",
                        "id": message.get("id"),
                        "error": {"code": -32601, "message": "Method not found"}
                    }
                    return JSONResponse(content=error_response, status_code=400)
                    
            except json.JSONDecodeError:
                return JSONResponse(content={"error": "Invalid JSON"}, status_code=400)
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                return JSONResponse(content={"error": str(e)}, status_code=500)
        
        @self.app.get("/")
        async def mcp_endpoint_get(
            request: Request,
            mcp_session_id: Optional[str] = Header(None, alias="Mcp-Session-Id"),
            accept: Optional[str] = Header(None)
        ):
            if accept and "text/event-stream" in accept:
                return StreamingResponse(
                    self._sse_stream(mcp_session_id),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
                )
            else:
                return JSONResponse(content={"error": "SSE not supported"}, status_code=405)
    
    def _is_allowed_origin(self, origin: str) -> bool:
        allowed_origins = ["http://localhost", "http://127.0.0.1", "https://localhost", "https://127.0.0.1"]
        return any(origin.startswith(allowed) for allowed in allowed_origins)
    
    async def _handle_initialize(self, message: Dict[str, Any], protocol_version: Optional[str]) -> JSONResponse:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "created_at": time.time(),
            "protocol_version": protocol_version or "2025-06-18"
        }
        
        response = {
            "jsonrpc": "2.0",
            "id": message.get("id"),
            "result": {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}, "logging": {}},
                "serverInfo": {"name": "Calculator MCP Server", "version": "1.0.0"}
            }
        }
        return JSONResponse(content=response, headers={"Mcp-Session-Id": session_id})
    
    async def _handle_list_tools(self, message: Dict[str, Any], session_id: Optional[str]) -> JSONResponse:
        if session_id and session_id not in self.sessions:
            return JSONResponse(content={"error": "Invalid session"}, status_code=404)
        
        response = {
            "jsonrpc": "2.0",
            "id": message.get("id"),
            "result": {"tools": self.tools}
        }
        return JSONResponse(content=response)
    
    async def _handle_call_tool(self, message: Dict[str, Any], session_id: Optional[str]) -> JSONResponse:
        if session_id and session_id not in self.sessions:
            return JSONResponse(content={"error": "Invalid session"}, status_code=404)
        
        params = message.get("params", {})
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        try:
            if tool_name == "add":
                result = await self._execute_add(arguments)
            elif tool_name == "subtract":
                result = await self._execute_subtract(arguments)
            elif tool_name == "multiply":
                result = await self._execute_multiply(arguments)
            elif tool_name == "divide":
                result = await self._execute_divide(arguments)
            elif tool_name == "power":
                result = await self._execute_power(arguments)
            elif tool_name == "sqrt":
                result = await self._execute_sqrt(arguments)
            elif tool_name == "percentage":
                result = await self._execute_percentage(arguments)
            elif tool_name == "compound_interest":
                result = await self._execute_compound_interest(arguments)
            elif tool_name == "statistics":
                result = await self._execute_statistics(arguments)
            elif tool_name == "health":
                result = await self._execute_health(arguments)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
            
            response = {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, ensure_ascii=False)
                        }
                    ]
                }
            }
            return JSONResponse(content=response)
            
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            error_response = {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "error": {"code": -32000, "message": str(e)}
            }
            return JSONResponse(content=error_response, status_code=500)
    
    async def _execute_add(self, arguments: Dict[str, Any]) -> Union[int, float]:
        a = arguments.get("a")
        b = arguments.get("b")
        result = a + b
        logger.info(f"Addition: {a} + {b} = {result}")
        return int(result) if isinstance(result, float) and result.is_integer() else round(result, 3)
    
    async def _execute_subtract(self, arguments: Dict[str, Any]) -> Union[int, float]:
        a = arguments.get("a")
        b = arguments.get("b")
        result = a - b
        logger.info(f"Subtraction: {a} - {b} = {result}")
        return int(result) if isinstance(result, float) and result.is_integer() else round(result, 3)
    
    async def _execute_multiply(self, arguments: Dict[str, Any]) -> Union[int, float]:
        a = arguments.get("a")
        b = arguments.get("b")
        result = a * b
        logger.info(f"Multiplication: {a} * {b} = {result}")
        return int(result) if isinstance(result, float) and result.is_integer() else round(result, 3)
    
    async def _execute_divide(self, arguments: Dict[str, Any]) -> Union[int, float]:
        a = arguments.get("a")
        b = arguments.get("b")
        if b == 0:
            raise ValueError("Cannot divide by zero")
        result = a / b
        logger.info(f"Division: {a} / {b} = {result}")
        return int(result) if isinstance(result, float) and result.is_integer() else round(result, 3)
    
    async def _execute_power(self, arguments: Dict[str, Any]) -> Union[int, float]:
        a = arguments.get("a")
        b = arguments.get("b")
        result = a ** b
        logger.info(f"Power: {a} ^ {b} = {result}")
        return int(result) if isinstance(result, float) and result.is_integer() else round(result, 3)
    
    async def _execute_sqrt(self, arguments: Dict[str, Any]) -> float:
        a = arguments.get("a")
        if a < 0:
            raise ValueError("Cannot calculate square root of negative number")
        result = math.sqrt(a)
        logger.info(f"Square root: √{a} = {result}")
        return round(result, 3)
    
    async def _execute_percentage(self, arguments: Dict[str, Any]) -> float:
        value = arguments.get("value")
        percent = arguments.get("percent")
        result = (value * percent) / 100
        logger.info(f"Percentage: {percent}% of {value} = {result}")
        return round(result, 2)
    
    async def _execute_compound_interest(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        principal = arguments.get("principal")
        rate = arguments.get("rate")
        time_years = arguments.get("time")
        frequency = arguments.get("frequency", 1)
        
        rate_decimal = rate / 100
        amount = principal * (1 + rate_decimal / frequency) ** (frequency * time_years)
        interest = amount - principal
        
        result = {
            "principal": round(principal, 2),
            "rate_percent": rate,
            "time_years": time_years,
            "compound_frequency": frequency,
            "final_amount": round(amount, 2),
            "interest_earned": round(interest, 2),
            "total_return_percent": round((interest / principal) * 100, 2)
        }
        logger.info(f"Compound interest calculated: {result}")
        return result
    
    async def _execute_statistics(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        numbers = arguments.get("numbers", [])
        if not numbers:
            raise ValueError("Cannot calculate statistics for empty list")
        
        nums = [float(n) for n in numbers]
        n = len(nums)
        total = sum(nums)
        mean = total / n
        variance = sum((x - mean) ** 2 for x in nums) / n
        std_dev = math.sqrt(variance)
        sorted_nums = sorted(nums)
        
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
    
    async def _execute_health(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "server": "Calculator Server",
            "timestamp": time.time(),
            "available_operations": 10
        }
    
    async def _sse_stream(self, session_id: Optional[str]):
        """Generate SSE stream for server-to-client communication"""
        yield f"data: {json.dumps({'type': 'connected', 'timestamp': time.time()})}\n\n"
        
        try:
            while True:
                await asyncio.sleep(30)
                yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': time.time()})}\n\n"
        except Exception as e:
            logger.error(f"SSE stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

# Create server instance
server = CalculatorMCPServer()
app = server.app

if __name__ == "__main__":
    print("=" * 60)
    print("🧮 Starting Calculator MCP Server")
    print("=" * 60)
    print(f"🌐 Port: 9104")
    print(f"⚡ Server Name: Calculator Server")
    print(f"🔧 Available Operations: 10")
    print(f"📋 MCP Protocol: 2025-06-18")
    print("=" * 60)
    
    uvicorn.run(app, host="127.0.0.1", port=9104, log_level="info")