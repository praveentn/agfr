# agentic/agents/local/tabulator_server.py
import json
import time
import uuid
import logging
import asyncio
from typing import Dict, List, Any, Union, Optional
import pandas as pd
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MCP Server Implementation
class TabulatorMCPServer:
    def __init__(self):
        self.app = FastAPI(title="Tabulator MCP Server")
        self.sessions = {}
        self.tools = [
            {
                "name": "tabulate",
                "description": "Convert input data into structured table format (JSON, CSV, or HTML)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "data": {"description": "Input data to tabulate (string, list, or dict)"},
                        "fields": {"type": "array", "items": {"type": "string"}, "description": "Specific fields to extract/display (optional)"},
                        "format": {"type": "string", "enum": ["json", "csv", "html"], "default": "json", "description": "Output format"}
                    },
                    "required": ["data"]
                }
            },
            {
                "name": "sort_data",
                "description": "Sort tabular data by specified field",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "data": {"description": "Input data to sort"},
                        "sort_field": {"type": "string", "description": "Field name to sort by"},
                        "ascending": {"type": "boolean", "default": True, "description": "Sort in ascending order"}
                    },
                    "required": ["data", "sort_field"]
                }
            },
            {
                "name": "filter_data",
                "description": "Filter tabular data based on field value",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "data": {"description": "Input data to filter"},
                        "filter_field": {"type": "string", "description": "Field to filter on"},
                        "filter_value": {"type": "string", "description": "Value to filter for"},
                        "operator": {"type": "string", "enum": ["contains", "equals", "starts_with", "greater_than", "less_than"], "default": "contains", "description": "Filter operator"}
                    },
                    "required": ["data", "filter_field", "filter_value"]
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
                "serverInfo": {"name": "Tabulator MCP Server", "version": "1.0.0"}
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
            if tool_name == "tabulate":
                result = await self._execute_tabulate(arguments)
            elif tool_name == "sort_data":
                result = await self._execute_sort_data(arguments)
            elif tool_name == "filter_data":
                result = await self._execute_filter_data(arguments)
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
    
    async def _execute_tabulate(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Convert input data into structured table format"""
        try:
            data = arguments.get("data")
            fields = arguments.get("fields")
            format_type = arguments.get("format", "json")
            
            logger.info(f"Tabulating data: type={type(data)}, format={format_type}")
            
            # Parse and normalize input data
            if isinstance(data, str):
                try:
                    parsed = json.loads(data)
                    logger.info("Parsed data as JSON")
                except json.JSONDecodeError:
                    parsed = self._extract_data_from_text(data, fields)
                    logger.info(f"Extracted {len(parsed)} items from text")
            else:
                parsed = data

            # Normalize to list of dictionaries
            table_data = self._normalize_to_table(parsed)
            logger.info(f"Normalized to {len(table_data)} table rows")

            # Filter and organize fields
            if fields:
                table_data = self._filter_fields(table_data, fields)
            else:
                fields = self._auto_detect_fields(table_data)

            # Apply format-specific processing
            if format_type.lower() == "csv":
                return self._format_as_csv(table_data, fields)
            elif format_type.lower() == "html":
                return self._format_as_html(table_data, fields)
            else:
                return self._format_as_json(table_data, fields)

        except Exception as e:
            logger.error(f"Tabulation error: {e}")
            return {
                "table": [],
                "error": str(e),
                "format": format_type,
                "row_count": 0,
                "columns": [],
                "success": False
            }
    
    async def _execute_sort_data(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Sort tabular data by specified field"""
        try:
            data = arguments.get("data")
            sort_field = arguments.get("sort_field")
            ascending = arguments.get("ascending", True)
            
            logger.info(f"Sorting data by field: {sort_field}, ascending: {ascending}")
            
            if isinstance(data, str):
                parsed_data = json.loads(data)
            else:
                parsed_data = data
            
            table_data = self._normalize_to_table(parsed_data)
            
            if not table_data:
                return {"table": [], "row_count": 0, "columns": [], "success": False}
            
            if sort_field in table_data[0]:
                sorted_data = sorted(
                    table_data, 
                    key=lambda x: x.get(sort_field, ""), 
                    reverse=not ascending
                )
            else:
                sorted_data = table_data
            
            return {
                "table": sorted_data,
                "format": "json",
                "row_count": len(sorted_data),
                "columns": list(sorted_data[0].keys()) if sorted_data else [],
                "sorted_by": sort_field,
                "ascending": ascending,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Sort failed: {e}")
            return {
                "table": [],
                "error": str(e),
                "format": "json",
                "row_count": 0,
                "success": False
            }
    
    async def _execute_filter_data(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Filter tabular data based on field value"""
        try:
            data = arguments.get("data")
            filter_field = arguments.get("filter_field")
            filter_value = arguments.get("filter_value")
            operator = arguments.get("operator", "contains")
            
            logger.info(f"Filtering data: field={filter_field}, value={filter_value}, operator={operator}")
            
            if isinstance(data, str):
                parsed_data = json.loads(data)
            else:
                parsed_data = data
            
            table_data = self._normalize_to_table(parsed_data)
            original_count = len(table_data)
            
            # Apply filter
            filtered_data = []
            for row in table_data:
                field_value = str(row.get(filter_field, "")).lower()
                filter_val_lower = str(filter_value).lower()
                
                match = False
                if operator == "contains":
                    match = filter_val_lower in field_value
                elif operator == "equals":
                    match = field_value == filter_val_lower
                elif operator == "starts_with":
                    match = field_value.startswith(filter_val_lower)
                elif operator == "greater_than":
                    try:
                        match = float(field_value) > float(filter_value)
                    except ValueError:
                        match = field_value > filter_val_lower
                elif operator == "less_than":
                    try:
                        match = float(field_value) < float(filter_value)
                    except ValueError:
                        match = field_value < filter_val_lower
                
                if match:
                    filtered_data.append(row)
            
            return {
                "table": filtered_data,
                "format": "json",
                "row_count": len(filtered_data),
                "columns": list(filtered_data[0].keys()) if filtered_data else [],
                "filter_applied": f"{filter_field} {operator} {filter_value}",
                "original_count": original_count,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Filter failed: {e}")
            return {
                "table": [],
                "error": str(e),
                "format": "json",
                "row_count": 0,
                "success": False
            }
    
    async def _execute_health(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute health check tool"""
        return {
            "status": "healthy",
            "server": "Tabulator Server",
            "timestamp": time.time(),
            "available_formats": ["json", "csv", "html"]
        }
    
    def _normalize_to_table(self, data: Any) -> List[Dict]:
        """Normalize various data formats to list of dictionaries"""
        if isinstance(data, dict):
            if "items" in data:
                return data["items"]
            elif "findings" in data:
                return data["findings"]
            elif "summary" in data and "key_points" in data:
                return [{
                    "type": "summary",
                    "content": data["summary"],
                    "key_points": ", ".join(data.get("key_points", [])),
                    "word_count": data.get("word_count", 0)
                }]
            else:
                return [data]
        elif isinstance(data, list):
            normalized = []
            for item in data:
                if isinstance(item, dict):
                    normalized.append(item)
                else:
                    normalized.append({"value": str(item)})
            return normalized
        else:
            return [{"value": str(data)}]
    
    def _extract_data_from_text(self, text: str, fields: List[str] = None) -> List[Dict]:
        """Extract structured data from plain text"""
        rows = []
        lines = text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if fields:
                if any(delim in line for delim in [',', ';', '|', '\t']):
                    for delim in [',', ';', '|', '\t']:
                        if delim in line:
                            parts = [p.strip() for p in line.split(delim)]
                            row = {}
                            for i, field in enumerate(fields):
                                row[field] = parts[i] if i < len(parts) else "N/A"
                            rows.append(row)
                            break
                elif ':' in line:
                    key, value = line.split(':', 1)
                    if len(fields) >= 2:
                        rows.append({fields[0]: key.strip(), fields[1]: value.strip()})
                    else:
                        rows.append({"key": key.strip(), "value": value.strip()})
                else:
                    if fields:
                        rows.append({fields[0]: line})
                    else:
                        rows.append({"content": line})
            else:
                if ':' in line and len(line.split(':')) == 2:
                    key, value = line.split(':', 1)
                    rows.append({"key": key.strip(), "value": value.strip()})
                elif any(delim in line for delim in [',', ';', '|']):
                    for delim in [',', ';', '|']:
                        if delim in line:
                            parts = [p.strip() for p in line.split(delim)]
                            if len(parts) >= 2:
                                row = {f"column_{i+1}": part for i, part in enumerate(parts)}
                                rows.append(row)
                            break
                else:
                    rows.append({"content": line})
        
        return rows
    
    def _auto_detect_fields(self, table_data: List[Dict]) -> List[str]:
        """Auto-detect important fields from data"""
        if not table_data:
            return []
        
        all_keys = set()
        for row in table_data:
            all_keys.update(row.keys())
        
        priority_fields = [
            "title", "name", "subject", "topic",
            "url", "link", "source", 
            "snippet", "description", "summary", "content",
            "date", "date_published", "timestamp",
            "price", "value", "amount", "score",
            "type", "category", "status"
        ]
        
        ordered_fields = []
        for field in priority_fields:
            if field in all_keys:
                ordered_fields.append(field)
                all_keys.remove(field)
        
        ordered_fields.extend(sorted(all_keys))
        return ordered_fields
    
    def _filter_fields(self, table_data: List[Dict], fields: List[str]) -> List[Dict]:
        """Filter data to only include specified fields"""
        filtered_data = []
        for row in table_data:
            filtered_row = {}
            for field in fields:
                value = row.get(field, "N/A")
                if isinstance(value, str):
                    value = value.strip()
                    if len(value) > 200:
                        value = value[:197] + "..."
                filtered_row[field] = value
            filtered_data.append(filtered_row)
        return filtered_data
    
    def _format_as_json(self, table_data: List[Dict], fields: List[str]) -> Dict[str, Any]:
        """Format data as JSON"""
        return {
            "table": table_data,
            "format": "json",
            "row_count": len(table_data),
            "columns": fields or (list(table_data[0].keys()) if table_data else []),
            "success": True
        }
    
    def _format_as_csv(self, table_data: List[Dict], fields: List[str]) -> Dict[str, Any]:
        """Format data as CSV string"""
        if not table_data:
            return {"table": "", "format": "csv", "row_count": 0, "columns": [], "success": True}
        
        df = pd.DataFrame(table_data)
        if fields:
            existing_fields = [f for f in fields if f in df.columns]
            missing_fields = [f for f in fields if f not in df.columns]
            
            for field in missing_fields:
                df[field] = "N/A"
            
            df = df[fields]
        
        csv_output = df.to_csv(index=False)
        
        return {
            "table": csv_output,
            "format": "csv",
            "row_count": len(table_data),
            "columns": list(df.columns),
            "success": True
        }
    
    def _format_as_html(self, table_data: List[Dict], fields: List[str]) -> Dict[str, Any]:
        """Format data as HTML table"""
        if not table_data:
            return {"table": "<p>No data to display</p>", "format": "html", "row_count": 0, "columns": [], "success": True}
        
        df = pd.DataFrame(table_data)
        if fields:
            existing_fields = [f for f in fields if f in df.columns]
            missing_fields = [f for f in fields if f not in df.columns]
            
            for field in missing_fields:
                df[field] = "N/A"
            
            df = df[fields]
        
        html_output = df.to_html(
            index=False, 
            classes="table table-striped table-hover",
            escape=False,
            border=0
        )
        
        return {
            "table": html_output,
            "format": "html", 
            "row_count": len(table_data),
            "columns": list(df.columns),
            "success": True
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
server = TabulatorMCPServer()
app = server.app

if __name__ == "__main__":
    print("=" * 60)
    print("📊 Starting Tabulator MCP Server")
    print("=" * 60)
    print(f"🌐 Port: 9102")
    print(f"⚡ Server Name: Tabulator Server")
    print(f"🔧 Available Formats: JSON, CSV, HTML")
    print(f"📋 MCP Protocol: 2025-06-18")
    print("=" * 60)
    
    uvicorn.run(app, host="127.0.0.1", port=9102, log_level="info")
