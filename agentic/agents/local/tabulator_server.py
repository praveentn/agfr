# agentic/agents/local/tabulator_server.py
import json
import time
import uuid
import re
import sys
import os
import logging
import asyncio
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import pandas as pd

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from agentic.core.llm_client import llm_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedTabulatorMCPServer:
    """Enhanced Tabulator with AI-powered data processing and intelligent structuring"""
    
    def __init__(self):
        self.app = FastAPI(title="Enhanced Tabulator MCP Server")
        self.sessions = {}
        self.tools = [
            {
                "name": "ai_tabulate",
                "description": "AI-powered intelligent data tabulation with automatic structure detection and enhancement",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": ["string", "array", "object"], "description": "Data to tabulate (text, JSON, CSV, or structured data)"},
                        "format": {"type": "string", "enum": ["auto", "csv", "json", "html", "markdown", "excel"], "default": "auto", "description": "Output format"},
                        "columns": {"type": "array", "items": {"type": "string"}, "description": "Specific columns to include (optional)"},
                        "ai_enhancement": {"type": "boolean", "default": True, "description": "Use AI to enhance and structure data"},
                        "style": {"type": "string", "enum": ["clean", "professional", "detailed", "compact"], "default": "professional", "description": "Table styling"},
                        "max_rows": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 100, "description": "Maximum number of rows"},
                        "sort_by": {"type": "string", "description": "Column to sort by (optional)"},
                        "group_by": {"type": "string", "description": "Column to group by (optional)"},
                        "include_summary": {"type": "boolean", "default": False, "description": "Include summary statistics"}
                    },
                    "required": ["data"]
                }
            },
            {
                "name": "structure_data",
                "description": "Structure unstructured text data into organized tables using AI",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Unstructured text to organize", "minLength": 10},
                        "target_structure": {"type": "string", "description": "Desired structure type", "examples": ["product_comparison", "financial_data", "contact_list", "event_schedule"]},
                        "extract_fields": {"type": "array", "items": {"type": "string"}, "description": "Specific fields to extract"},
                        "output_format": {"type": "string", "enum": ["table", "json", "csv"], "default": "table", "description": "Output format"},
                        "confidence_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.7, "description": "Minimum confidence for data extraction"}
                    },
                    "required": ["text"]
                }
            },
            {
                "name": "smart_filter",
                "description": "Intelligently filter and query tabular data using natural language",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "table_data": {"type": ["array", "object"], "description": "Table data to filter"},
                        "filter_criteria": {"type": "string", "description": "Natural language filter criteria", "examples": ["top 10 by revenue", "products under $100", "items from last month"]},
                        "filter_type": {"type": "string", "enum": ["smart", "basic", "advanced"], "default": "smart", "description": "Type of filtering to apply"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 50, "description": "Maximum results to return"},
                        "output_explanation": {"type": "boolean", "default": True, "description": "Include explanation of filtering logic"}
                    },
                    "required": ["table_data", "filter_criteria"]
                }
            },
            {
                "name": "merge_tables",
                "description": "Intelligently merge multiple tables with AI-assisted matching",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "tables": {"type": "array", "items": {"type": "object"}, "minItems": 2, "description": "List of tables to merge"},
                        "merge_strategy": {"type": "string", "enum": ["auto", "inner", "outer", "left", "right"], "default": "auto", "description": "Merge strategy"},
                        "match_columns": {"type": "array", "items": {"type": "string"}, "description": "Columns to match on (optional, AI will suggest if not provided)"},
                        "resolve_conflicts": {"type": "string", "enum": ["ai_decide", "keep_first", "keep_last", "combine"], "default": "ai_decide", "description": "How to resolve conflicting data"},
                        "include_metadata": {"type": "boolean", "default": True, "description": "Include merge metadata and statistics"}
                    },
                    "required": ["tables"]
                }
            },
            {
                "name": "analyze_table",
                "description": "Analyze table data and provide insights, statistics, and recommendations",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "table_data": {"type": ["array", "object"], "description": "Table data to analyze"},
                        "analysis_type": {"type": "string", "enum": ["descriptive", "comparative", "trend", "quality", "comprehensive"], "default": "comprehensive", "description": "Type of analysis to perform"},
                        "focus_columns": {"type": "array", "items": {"type": "string"}, "description": "Specific columns to focus analysis on"},
                        "include_visualizations": {"type": "boolean", "default": False, "description": "Include visualization suggestions"},
                        "business_context": {"type": "string", "description": "Business context for analysis"}
                    },
                    "required": ["table_data"]
                }
            },
            {
                "name": "export_table",
                "description": "Export table data in various formats with customizable options",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "table_data": {"type": ["array", "object"], "description": "Table data to export"},
                        "export_format": {"type": "string", "enum": ["csv", "json", "html", "markdown", "excel", "pdf"], "default": "csv", "description": "Export format"},
                        "include_headers": {"type": "boolean", "default": True, "description": "Include column headers"},
                        "custom_styling": {"type": "object", "description": "Custom styling options"},
                        "filename": {"type": "string", "description": "Suggested filename"}
                    },
                    "required": ["table_data"]
                }
            }
        ]
        
        self._setup_routes()
        self._setup_middleware()
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/mcp/info")
        async def get_server_info():
            """Get server information"""
            return {
                "name": "Enhanced Tabulator Server",
                "version": "2.0.0",
                "description": "AI-powered data tabulation and analysis",
                "capabilities": [
                    "ai_tabulation", "data_structuring", "smart_filtering",
                    "table_merging", "data_analysis", "multi_format_export"
                ],
                "protocol": "mcp-2025-06-18",
                "llm_integration": "azure_openai" if llm_client and llm_client.client else "fallback"
            }
        
        @self.app.get("/mcp/tools")
        async def list_tools():
            """List available tools"""
            return {"tools": self.tools}
        
        @self.app.post("/mcp/call")
        async def call_tool(request: Request):
            """Call a specific tool"""
            try:
                data = await request.json()
                tool_name = data.get("name")
                arguments = data.get("arguments", {})
                
                if tool_name not in [tool["name"] for tool in self.tools]:
                    raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
                
                # Route to appropriate handler
                if tool_name == "ai_tabulate":
                    result = await self._ai_tabulate(arguments)
                elif tool_name == "structure_data":
                    result = await self._structure_data(arguments)
                elif tool_name == "smart_filter":
                    result = await self._smart_filter(arguments)
                elif tool_name == "merge_tables":
                    result = await self._merge_tables(arguments)
                elif tool_name == "analyze_table":
                    result = await self._analyze_table(arguments)
                elif tool_name == "export_table":
                    result = await self._export_table(arguments)
                else:
                    raise HTTPException(status_code=400, detail=f"Tool '{tool_name}' not implemented")
                
                return {"success": True, "result": result}
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Tool execution failed: {e}")
                return {"success": False, "error": str(e)}
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            try:
                # Test Azure OpenAI connection
                llm_status = "healthy" if llm_client and llm_client.client else "not_configured"
                
                return {
                    "status": "healthy",
                    "server": "Enhanced Tabulator Server",
                    "azure_openai": llm_status,
                    "timestamp": time.time(),
                    "capabilities": [
                        "ai_tabulate", "structure_data", "smart_filter",
                        "merge_tables", "analyze_table", "export_table"
                    ]
                }
                
            except Exception as e:
                return {
                    "status": "error",
                    "error": str(e),
                    "timestamp": time.time()
                }
        
        @self.app.get("/events")
        async def event_stream(session_id: Optional[str] = None):
            """Server-sent events stream"""
            return StreamingResponse(
                self._sse_stream(session_id), 
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
    
    def _setup_middleware(self):
        """Setup middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Core Tabulation Tools with AI Integration
    
    async def _ai_tabulate(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """AI-powered intelligent data tabulation"""
        try:
            data = arguments.get("data")
            format_type = arguments.get("format", "auto")
            columns = arguments.get("columns", [])
            ai_enhancement = arguments.get("ai_enhancement", True)
            style = arguments.get("style", "professional")
            max_rows = arguments.get("max_rows", 100)
            sort_by = arguments.get("sort_by")
            group_by = arguments.get("group_by")
            include_summary = arguments.get("include_summary", False)
            
            if not data:
                return {"success": False, "error": "Data is required for tabulation"}
            
            # Parse and normalize data
            parsed_data = await self._parse_input_data(data)
            if not parsed_data["success"]:
                return parsed_data
            
            table_data = parsed_data["data"]
            
            # Apply AI enhancement if requested
            if ai_enhancement and llm_client and llm_client.client:
                enhanced_data = await self._ai_enhance_table(table_data, columns, style)
                if enhanced_data["success"]:
                    table_data = enhanced_data["data"]
                    if not columns:
                        columns = enhanced_data.get("suggested_columns", [])
            
            # Apply filtering and sorting
            if sort_by:
                table_data = self._sort_table_data(table_data, sort_by)
            
            if group_by:
                table_data = self._group_table_data(table_data, group_by)
            
            # Limit rows
            if len(table_data) > max_rows:
                table_data = table_data[:max_rows]
            
            # Format output
            formatted_result = await self._format_table_output(table_data, format_type, style, columns)
            
            result = {
                "success": True,
                "table": formatted_result["table"],
                "metadata": {
                    "rows": len(table_data),
                    "columns": len(columns) if columns else len(table_data[0].keys()) if table_data else 0,
                    "format": format_type,
                    "style": style,
                    "ai_enhanced": ai_enhancement,
                    "sorted_by": sort_by,
                    "grouped_by": group_by
                },
                "original_rows": parsed_data.get("original_rows", len(table_data)),
                "data_types": self._analyze_data_types(table_data)
            }
            
            # Add summary if requested
            if include_summary:
                summary = await self._generate_table_summary(table_data)
                result["summary"] = summary
            
            return result
            
        except Exception as e:
            logger.error(f"AI tabulation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _structure_data(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Structure unstructured text data using AI"""
        try:
            text = arguments.get("text", "").strip()
            target_structure = arguments.get("target_structure", "")
            extract_fields = arguments.get("extract_fields", [])
            output_format = arguments.get("output_format", "table")
            confidence_threshold = arguments.get("confidence_threshold", 0.7)
            
            if not text:
                return {"success": False, "error": "Text is required for data structuring"}
            
            # Use AI to structure data if available
            if llm_client and llm_client.client:
                return await self._ai_structure_data(text, target_structure, extract_fields, output_format, confidence_threshold)
            else:
                return await self._fallback_structure_data(text, extract_fields, output_format)
                
        except Exception as e:
            logger.error(f"Data structuring failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _smart_filter(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Smart filtering using natural language and AI"""
        try:
            table_data = arguments.get("table_data", [])
            filter_criteria = arguments.get("filter_criteria", "")
            filter_type = arguments.get("filter_type", "smart")
            limit = arguments.get("limit", 50)
            output_explanation = arguments.get("output_explanation", True)
            
            if not table_data or not filter_criteria:
                return {"success": False, "error": "Table data and filter criteria are required"}
            
            # Normalize table data
            if isinstance(table_data, dict):
                table_data = [table_data]
            
            # Use AI-powered filtering if available and requested
            if filter_type == "smart" and llm_client and llm_client.client:
                return await self._ai_smart_filter(table_data, filter_criteria, limit, output_explanation)
            else:
                return await self._basic_filter(table_data, filter_criteria, filter_type, limit)
                
        except Exception as e:
            logger.error(f"Smart filtering failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _merge_tables(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple tables with AI assistance"""
        try:
            tables = arguments.get("tables", [])
            merge_strategy = arguments.get("merge_strategy", "auto")
            match_columns = arguments.get("match_columns", [])
            resolve_conflicts = arguments.get("resolve_conflicts", "ai_decide")
            include_metadata = arguments.get("include_metadata", True)
            
            if len(tables) < 2:
                return {"success": False, "error": "At least 2 tables are required for merging"}
            
            # Use AI-assisted merging if available
            if llm_client and llm_client.client:
                return await self._ai_merge_tables(tables, merge_strategy, match_columns, resolve_conflicts, include_metadata)
            else:
                return await self._basic_merge_tables(tables, match_columns, merge_strategy)
                
        except Exception as e:
            logger.error(f"Table merging failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _analyze_table(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze table data and provide insights"""
        try:
            table_data = arguments.get("table_data", [])
            analysis_type = arguments.get("analysis_type", "comprehensive")
            focus_columns = arguments.get("focus_columns", [])
            include_visualizations = arguments.get("include_visualizations", False)
            business_context = arguments.get("business_context", "")
            
            if not table_data:
                return {"success": False, "error": "Table data is required for analysis"}
            
            # Normalize table data
            if isinstance(table_data, dict):
                table_data = [table_data]
            
            # Use AI-powered analysis if available
            if llm_client and llm_client.client:
                return await self._ai_analyze_table(table_data, analysis_type, focus_columns, include_visualizations, business_context)
            else:
                return await self._basic_analyze_table(table_data, analysis_type, focus_columns)
                
        except Exception as e:
            logger.error(f"Table analysis failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _export_table(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Export table data in various formats"""
        try:
            table_data = arguments.get("table_data", [])
            export_format = arguments.get("export_format", "csv")
            include_headers = arguments.get("include_headers", True)
            custom_styling = arguments.get("custom_styling", {})
            filename = arguments.get("filename", f"table_export_{int(time.time())}")
            
            if not table_data:
                return {"success": False, "error": "Table data is required for export"}
            
            # Normalize table data
            if isinstance(table_data, dict):
                table_data = [table_data]
            
            # Export in requested format
            exported_data = await self._export_in_format(table_data, export_format, include_headers, custom_styling, filename)
            
            return {
                "success": True,
                "exported_data": exported_data["content"],
                "format": export_format,
                "filename": filename,
                "size_bytes": len(exported_data["content"]),
                "rows_exported": len(table_data),
                "metadata": exported_data.get("metadata", {})
            }
            
        except Exception as e:
            logger.error(f"Table export failed: {e}")
            return {"success": False, "error": str(e)}
    
    # AI-Powered Implementation Methods
    
    async def _ai_enhance_table(self, table_data: List[Dict], columns: List[str], style: str) -> Dict[str, Any]:
        """AI-enhanced table processing"""
        try:
            # Sample data for analysis
            sample_data = table_data[:5] if len(table_data) > 5 else table_data
            
            prompt = f"""
            Analyze this table data and suggest improvements:
            
            Sample data: {json.dumps(sample_data, indent=2)}
            Requested columns: {columns}
            Style: {style}
            
            Please suggest:
            1. Optimal column order and names
            2. Data type conversions needed
            3. Missing or derived columns that would be valuable
            4. Data cleaning recommendations
            5. Formatting improvements
            
            Return as JSON:
            {{
                "suggested_columns": ["col1", "col2", "col3"],
                "column_mappings": {{"old_name": "new_name"}},
                "data_improvements": ["improvement1", "improvement2"],
                "derived_columns": {{"new_col": "calculation_description"}},
                "formatting_rules": {{"column": "format_rule"}}
            }}
            """
            
            messages = [
                {"role": "system", "content": "You are an expert data analyst. Suggest table improvements for better clarity and usability."},
                {"role": "user", "content": prompt}
            ]
            
            response = await llm_client.generate(messages, temperature=0.3, max_tokens=800)
            
            if response:
                try:
                    suggestions = json.loads(response.strip())
                    
                    # Apply suggestions to enhance the table
                    enhanced_data = self._apply_ai_suggestions(table_data, suggestions)
                    
                    return {
                        "success": True,
                        "data": enhanced_data,
                        "suggestions": suggestions,
                        "suggested_columns": suggestions.get("suggested_columns", [])
                    }
                    
                except json.JSONDecodeError:
                    return {"success": False, "error": "Failed to parse AI suggestions"}
            else:
                return {"success": False, "error": "No AI response received"}
                
        except Exception as e:
            logger.error(f"AI table enhancement failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _ai_structure_data(self, text: str, target_structure: str, extract_fields: List[str], output_format: str, confidence_threshold: float) -> Dict[str, Any]:
        """AI-powered data structuring"""
        try:
            field_instruction = f" Focus on extracting: {', '.join(extract_fields)}." if extract_fields else ""
            structure_instruction = f" Target structure: {target_structure}." if target_structure else ""
            
            prompt = f"""
            Extract structured data from the following text and organize it into a table.
            {structure_instruction}{field_instruction}
            
            Text: {text}
            
            Create a structured table with appropriate columns and rows.
            Include confidence scores for each extracted data point.
            
            Return as JSON:
            {{
                "structured_data": [
                    {{"column1": "value1", "column2": "value2", "confidence": 0.95}},
                    {{"column1": "value3", "column2": "value4", "confidence": 0.87}}
                ],
                "columns": ["column1", "column2"],
                "extraction_summary": {{
                    "total_entries": 2,
                    "high_confidence": 1,
                    "medium_confidence": 1,
                    "low_confidence": 0
                }},
                "methodology": "Brief description of extraction approach"
            }}
            """
            
            messages = [
                {"role": "system", "content": "You are an expert data extraction specialist. Extract and structure data accurately with confidence scores."},
                {"role": "user", "content": prompt}
            ]
            
            response = await llm_client.generate(messages, temperature=0.2, max_tokens=1200)
            
            if response:
                try:
                    result = json.loads(response.strip())
                    
                    # Filter by confidence threshold
                    structured_data = result.get("structured_data", [])
                    filtered_data = [
                        {k: v for k, v in item.items() if k != "confidence"}
                        for item in structured_data
                        if item.get("confidence", 0) >= confidence_threshold
                    ]
                    
                    # Format according to output_format
                    if output_format == "json":
                        formatted_output = json.dumps(filtered_data, indent=2)
                    elif output_format == "csv":
                        formatted_output = self._convert_to_csv(filtered_data)
                    else:  # table
                        formatted_output = self._format_as_table(filtered_data)
                    
                    return {
                        "success": True,
                        "structured_data": filtered_data,
                        "formatted_output": formatted_output,
                        "columns": result.get("columns", []),
                        "extraction_summary": result.get("extraction_summary", {}),
                        "methodology": result.get("methodology", ""),
                        "confidence_threshold": confidence_threshold,
                        "total_extracted": len(structured_data),
                        "filtered_count": len(filtered_data)
                    }
                    
                except json.JSONDecodeError:
                    return await self._fallback_structure_data(text, extract_fields, output_format)
            else:
                return await self._fallback_structure_data(text, extract_fields, output_format)
                
        except Exception as e:
            logger.error(f"AI data structuring failed: {e}")
            return await self._fallback_structure_data(text, extract_fields, output_format)
    
    async def _ai_smart_filter(self, table_data: List[Dict], filter_criteria: str, limit: int, output_explanation: bool) -> Dict[str, Any]:
        """AI-powered smart filtering"""
        try:
            # Sample data for analysis
            sample_data = table_data[:3] if len(table_data) > 3 else table_data
            available_columns = list(table_data[0].keys()) if table_data else []
            
            prompt = f"""
            Apply smart filtering to this table data based on the criteria.
            
            Sample data structure: {json.dumps(sample_data, indent=2)}
            Available columns: {available_columns}
            Filter criteria: "{filter_criteria}"
            
            Analyze the criteria and determine:
            1. Which columns to filter on
            2. What conditions to apply
            3. How to interpret the natural language request
            4. Appropriate sorting/ranking if needed
            
            Return filtering logic as JSON:
            {{
                "filter_logic": {{
                    "column": "column_name",
                    "operation": "equals/contains/greater_than/less_than/range",
                    "value": "filter_value",
                    "additional_filters": []
                }},
                "sort_logic": {{
                    "column": "sort_column",
                    "direction": "asc/desc"
                }},
                "explanation": "Clear explanation of filtering logic",
                "confidence": 0.9
            }}
            """
            
            messages = [
                {"role": "system", "content": "You are an expert data filtering specialist. Translate natural language into precise filtering logic."},
                {"role": "user", "content": prompt}
            ]
            
            response = await llm_client.generate(messages, temperature=0.2, max_tokens=600)
            
            if response:
                try:
                    filter_logic = json.loads(response.strip())
                    
                    # Apply the filtering logic
                    filtered_data = self._apply_ai_filter_logic(table_data, filter_logic)
                    
                    # Limit results
                    limited_data = filtered_data[:limit]
                    
                    result = {
                        "success": True,
                        "filtered_data": limited_data,
                        "total_matches": len(filtered_data),
                        "returned_count": len(limited_data),
                        "filter_criteria": filter_criteria,
                        "confidence": filter_logic.get("confidence", 0.8)
                    }
                    
                    if output_explanation:
                        result["explanation"] = filter_logic.get("explanation", "AI-based filtering applied")
                        result["filter_logic"] = filter_logic.get("filter_logic", {})
                    
                    return result
                    
                except json.JSONDecodeError:
                    return await self._basic_filter(table_data, filter_criteria, "basic", limit)
            else:
                return await self._basic_filter(table_data, filter_criteria, "basic", limit)
                
        except Exception as e:
            logger.error(f"AI smart filtering failed: {e}")
            return await self._basic_filter(table_data, filter_criteria, "basic", limit)
    
    async def _ai_merge_tables(self, tables: List[Dict], merge_strategy: str, match_columns: List[str], resolve_conflicts: str, include_metadata: bool) -> Dict[str, Any]:
        """AI-assisted table merging"""
        try:
            # Analyze table structures
            table_analyses = []
            for i, table in enumerate(tables):
                if isinstance(table, list) and table:
                    sample = table[0] if table else {}
                    table_analyses.append({
                        "table_index": i,
                        "columns": list(sample.keys()),
                        "row_count": len(table),
                        "sample_data": sample
                    })
            
            prompt = f"""
            Analyze these tables and suggest the best merge strategy:
            
            Tables: {json.dumps(table_analyses, indent=2)}
            Requested strategy: {merge_strategy}
            Suggested match columns: {match_columns}
            
            Determine:
            1. Best columns to join on (if auto strategy)
            2. Optimal merge type (inner/outer/left/right)
            3. How to handle column name conflicts
            4. Data type compatibility issues
            
            Return as JSON:
            {{
                "recommended_strategy": "inner/outer/left/right",
                "join_columns": ["column1", "column2"],
                "column_mappings": {{"table1_col": "unified_name"}},
                "conflict_resolution": "strategy for handling conflicts",
                "merge_plan": "step by step merge approach",
                "expected_result_size": "estimated rows",
                "confidence": 0.9
            }}
            """
            
            messages = [
                {"role": "system", "content": "You are an expert database specialist. Design optimal table merge strategies."},
                {"role": "user", "content": prompt}
            ]
            
            response = await llm_client.generate(messages, temperature=0.2, max_tokens=800)
            
            if response:
                try:
                    merge_plan = json.loads(response.strip())
                    
                    # Execute the merge based on AI recommendations
                    merged_result = self._execute_ai_merge(tables, merge_plan, resolve_conflicts)
                    
                    result = {
                        "success": True,
                        "merged_table": merged_result["data"],
                        "merge_strategy": merge_plan.get("recommended_strategy", merge_strategy),
                        "join_columns": merge_plan.get("join_columns", match_columns),
                        "total_rows": len(merged_result["data"]),
                        "confidence": merge_plan.get("confidence", 0.8)
                    }
                    
                    if include_metadata:
                        result["metadata"] = {
                            "original_tables": len(tables),
                            "merge_plan": merge_plan.get("merge_plan", ""),
                            "conflict_resolution": resolve_conflicts,
                            "column_mappings": merge_plan.get("column_mappings", {}),
                            "merge_statistics": merged_result.get("statistics", {})
                        }
                    
                    return result
                    
                except json.JSONDecodeError:
                    return await self._basic_merge_tables(tables, match_columns, merge_strategy)
            else:
                return await self._basic_merge_tables(tables, match_columns, merge_strategy)
                
        except Exception as e:
            logger.error(f"AI table merging failed: {e}")
            return await self._basic_merge_tables(tables, match_columns, merge_strategy)
    
    async def _ai_analyze_table(self, table_data: List[Dict], analysis_type: str, focus_columns: List[str], include_visualizations: bool, business_context: str) -> Dict[str, Any]:
        """AI-powered table analysis"""
        try:
            # Sample data for analysis
            sample_data = table_data[:5] if len(table_data) > 5 else table_data
            data_types = self._analyze_data_types(table_data)
            
            context_instruction = f" Business context: {business_context}." if business_context else ""
            focus_instruction = f" Focus analysis on: {', '.join(focus_columns)}." if focus_columns else ""
            viz_instruction = " Include visualization recommendations." if include_visualizations else ""
            
            prompt = f"""
            Perform {analysis_type} analysis on this table data.
            {context_instruction}{focus_instruction}{viz_instruction}
            
            Sample data: {json.dumps(sample_data, indent=2)}
            Data types: {data_types}
            Total rows: {len(table_data)}
            
            Provide comprehensive analysis including:
            1. Key statistics and metrics
            2. Data quality assessment
            3. Patterns and trends identified
            4. Insights and recommendations
            5. Potential issues or anomalies
            
            Return as JSON:
            {{
                "summary": "Executive summary of findings",
                "key_metrics": {{"metric1": "value1", "metric2": "value2"}},
                "insights": ["insight1", "insight2", "insight3"],
                "data_quality": {{
                    "completeness": 0.95,
                    "consistency": 0.88,
                    "issues": ["issue1", "issue2"]
                }},
                "recommendations": ["rec1", "rec2"],
                "visualization_suggestions": ["chart1", "chart2"],
                "confidence": 0.9
            }}
            """
            
            messages = [
                {"role": "system", "content": f"You are an expert data analyst specializing in {business_context if business_context else 'general business analysis'}. Provide actionable insights."},
                {"role": "user", "content": prompt}
            ]
            
            response = await llm_client.generate(messages, temperature=0.3, max_tokens=1200)
            
            if response:
                try:
                    analysis = json.loads(response.strip())
                    
                    return {
                        "success": True,
                        "analysis_type": analysis_type,
                        "summary": analysis.get("summary", ""),
                        "key_metrics": analysis.get("key_metrics", {}),
                        "insights": analysis.get("insights", []),
                        "data_quality": analysis.get("data_quality", {}),
                        "recommendations": analysis.get("recommendations", []),
                        "visualization_suggestions": analysis.get("visualization_suggestions", []) if include_visualizations else [],
                        "confidence": analysis.get("confidence", 0.8),
                        "business_context": business_context,
                        "focus_columns": focus_columns,
                        "rows_analyzed": len(table_data),
                        "columns_analyzed": len(data_types)
                    }
                    
                except json.JSONDecodeError:
                    return await self._basic_analyze_table(table_data, analysis_type, focus_columns)
            else:
                return await self._basic_analyze_table(table_data, analysis_type, focus_columns)
                
        except Exception as e:
            logger.error(f"AI table analysis failed: {e}")
            return await self._basic_analyze_table(table_data, analysis_type, focus_columns)
    
    # Helper and Utility Methods
    
    async def _parse_input_data(self, data: Any) -> Dict[str, Any]:
        """Parse and normalize input data to consistent format"""
        try:
            if isinstance(data, str):
                # Try to parse as JSON first
                try:
                    json_data = json.loads(data)
                    if isinstance(json_data, list):
                        return {"success": True, "data": json_data, "original_rows": len(json_data)}
                    elif isinstance(json_data, dict):
                        return {"success": True, "data": [json_data], "original_rows": 1}
                except json.JSONDecodeError:
                    # Try to parse as CSV
                    csv_data = self._parse_csv_string(data)
                    if csv_data:
                        return {"success": True, "data": csv_data, "original_rows": len(csv_data)}
                    
                    # Fallback: treat as unstructured text
                    return {"success": False, "error": "Unable to parse string data as JSON or CSV"}
            
            elif isinstance(data, list):
                return {"success": True, "data": data, "original_rows": len(data)}
            
            elif isinstance(data, dict):
                return {"success": True, "data": [data], "original_rows": 1}
            
            else:
                return {"success": False, "error": f"Unsupported data type: {type(data)}"}
                
        except Exception as e:
            return {"success": False, "error": f"Data parsing failed: {str(e)}"}
    
    def _parse_csv_string(self, csv_string: str) -> Optional[List[Dict[str, Any]]]:
        """Parse CSV string into list of dictionaries"""
        try:
            lines = csv_string.strip().split('\n')
            if len(lines) < 2:
                return None
            
            headers = [h.strip() for h in lines[0].split(',')]
            data = []
            
            for line in lines[1:]:
                values = [v.strip() for v in line.split(',')]
                if len(values) == len(headers):
                    row = dict(zip(headers, values))
                    data.append(row)
            
            return data if data else None
            
        except Exception:
            return None
    
    def _analyze_data_types(self, table_data: List[Dict]) -> Dict[str, str]:
        """Analyze data types in table"""
        if not table_data:
            return {}
        
        type_analysis = {}
        sample_row = table_data[0]
        
        for field, value in sample_row.items():
            if isinstance(value, (int, float)):
                type_analysis[field] = "numeric"
            elif isinstance(value, bool):
                type_analysis[field] = "boolean"
            elif isinstance(value, str):
                if re.match(r'\d{4}-\d{2}-\d{2}', value):
                    type_analysis[field] = "date"
                elif re.match(r'^https?://', value):
                    type_analysis[field] = "url"
                elif value.replace('.', '').replace('-', '').isdigit():
                    type_analysis[field] = "numeric_string"
                else:
                    type_analysis[field] = "text"
            else:
                type_analysis[field] = "unknown"
        
        return type_analysis
    
    def _sort_table_data(self, table_data: List[Dict], sort_by: str) -> List[Dict]:
        """Sort table data by specified column"""
        try:
            if not table_data or sort_by not in table_data[0]:
                return table_data
            
            # Determine sort direction
            reverse = False
            if sort_by.startswith('-'):
                reverse = True
                sort_by = sort_by[1:]
            
            return sorted(table_data, key=lambda x: x.get(sort_by, ''), reverse=reverse)
            
        except Exception as e:
            logger.warning(f"Sorting failed: {e}")
            return table_data
    
    def _group_table_data(self, table_data: List[Dict], group_by: str) -> List[Dict]:
        """Group table data by specified column"""
        try:
            if not table_data or group_by not in table_data[0]:
                return table_data
            
            grouped = {}
            for row in table_data:
                key = row.get(group_by, 'Unknown')
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append(row)
            
            # Flatten grouped data with group indicators
            result = []
            for group_key, group_rows in grouped.items():
                # Add group header
                group_header = {col: f"=== {group_key} ===" if col == group_by else "" for col in group_rows[0].keys()}
                result.append(group_header)
                result.extend(group_rows)
            
            return result
            
        except Exception as e:
            logger.warning(f"Grouping failed: {e}")
            return table_data
    
    async def _format_table_output(self, table_data: List[Dict], format_type: str, style: str, columns: List[str]) -> Dict[str, Any]:
        """Format table data according to specified format"""
        try:
            if not table_data:
                return {"table": "", "format": format_type}
            
            # Filter columns if specified
            if columns:
                filtered_data = []
                for row in table_data:
                    filtered_row = {col: row.get(col, '') for col in columns}
                    filtered_data.append(filtered_row)
                table_data = filtered_data
            
            if format_type == "auto":
                format_type = "markdown"  # Default to markdown for readability
            
            if format_type == "json":
                formatted = json.dumps(table_data, indent=2)
            elif format_type == "csv":
                formatted = self._convert_to_csv(table_data)
            elif format_type == "html":
                formatted = self._convert_to_html(table_data, style)
            elif format_type == "markdown":
                formatted = self._convert_to_markdown(table_data)
            elif format_type == "excel":
                formatted = self._convert_to_excel_info(table_data)
            else:
                formatted = self._format_as_table(table_data)
            
            return {"table": formatted, "format": format_type}
            
        except Exception as e:
            logger.error(f"Table formatting failed: {e}")
            return {"table": str(table_data), "format": "raw"}
    
    def _convert_to_csv(self, table_data: List[Dict]) -> str:
        """Convert table data to CSV format"""
        if not table_data:
            return ""
        
        headers = list(table_data[0].keys())
        csv_lines = [','.join(headers)]
        
        for row in table_data:
            values = [str(row.get(header, '')).replace(',', ';') for header in headers]
            csv_lines.append(','.join(values))
        
        return '\n'.join(csv_lines)
    
    def _convert_to_markdown(self, table_data: List[Dict]) -> str:
        """Convert table data to Markdown table format"""
        if not table_data:
            return ""
        
        headers = list(table_data[0].keys())
        
        # Header row
        markdown_lines = ['| ' + ' | '.join(headers) + ' |']
        
        # Separator row
        separator = '| ' + ' | '.join(['---'] * len(headers)) + ' |'
        markdown_lines.append(separator)
        
        # Data rows
        for row in table_data:
            values = [str(row.get(header, '')).replace('|', '\\|') for header in headers]
            markdown_lines.append('| ' + ' | '.join(values) + ' |')
        
        return '\n'.join(markdown_lines)
    
    def _convert_to_html(self, table_data: List[Dict], style: str) -> str:
        """Convert table data to HTML table format"""
        if not table_data:
            return "<table></table>"
        
        style_classes = {
            "clean": "table-clean",
            "professional": "table-professional",
            "detailed": "table-detailed",
            "compact": "table-compact"
        }
        
        css_class = style_classes.get(style, "table-professional")
        headers = list(table_data[0].keys())
        
        html = f'<table class="{css_class}">\n'
        
        # Header
        html += '  <thead>\n    <tr>\n'
        for header in headers:
            html += f'      <th>{header}</th>\n'
        html += '    </tr>\n  </thead>\n'
        
        # Body
        html += '  <tbody>\n'
        for row in table_data:
            html += '    <tr>\n'
            for header in headers:
                value = str(row.get(header, ''))
                html += f'      <td>{value}</td>\n'
            html += '    </tr>\n'
        html += '  </tbody>\n'
        
        html += '</table>'
        return html
    
    def _format_as_table(self, table_data: List[Dict]) -> str:
        """Format as plain text table"""
        if not table_data:
            return ""
        
        headers = list(table_data[0].keys())
        
        # Calculate column widths
        col_widths = {}
        for header in headers:
            col_widths[header] = len(header)
            for row in table_data:
                value_len = len(str(row.get(header, '')))
                if value_len > col_widths[header]:
                    col_widths[header] = value_len
        
        # Format table
        lines = []
        
        # Header
        header_line = '| ' + ' | '.join(header.ljust(col_widths[header]) for header in headers) + ' |'
        lines.append(header_line)
        
        # Separator
        separator = '| ' + ' | '.join('-' * col_widths[header] for header in headers) + ' |'
        lines.append(separator)
        
        # Data rows
        for row in table_data:
            values = [str(row.get(header, '')).ljust(col_widths[header]) for header in headers]
            data_line = '| ' + ' | '.join(values) + ' |'
            lines.append(data_line)
        
        return '\n'.join(lines)
    
    async def _generate_table_summary(self, table_data: List[Dict]) -> Dict[str, Any]:
        """Generate summary statistics for table"""
        try:
            if not table_data:
                return {"rows": 0, "columns": 0}
            
            summary = {
                "rows": len(table_data),
                "columns": len(table_data[0].keys()),
                "column_names": list(table_data[0].keys()),
                "data_types": self._analyze_data_types(table_data)
            }
            
            # Calculate basic statistics for numeric columns
            numeric_stats = {}
            for col in summary["column_names"]:
                values = [row.get(col) for row in table_data if row.get(col) is not None]
                
                # Try to convert to numeric
                numeric_values = []
                for val in values:
                    try:
                        numeric_values.append(float(val))
                    except (ValueError, TypeError):
                        continue
                
                if numeric_values:
                    numeric_stats[col] = {
                        "min": round(min(numeric_values), 3),
                        "max": round(max(numeric_values), 3),
                        "avg": round(sum(numeric_values) / len(numeric_values), 3),
                        "count": len(numeric_values)
                    }
            
            if numeric_stats:
                summary["numeric_statistics"] = numeric_stats
            
            return summary
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return {"error": str(e)}
    
    # Fallback implementations for when AI is not available
    
    async def _fallback_structure_data(self, text: str, extract_fields: List[str], output_format: str) -> Dict[str, Any]:
        """Fallback data structuring using regex patterns"""
        try:
            # Simple pattern-based extraction
            lines = text.split('\n')
            structured_data = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Try to extract key-value pairs
                if ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        structured_data.append({"field": key, "value": value})
                
                # Try to extract comma-separated values
                elif ',' in line:
                    values = [v.strip() for v in line.split(',')]
                    if len(values) >= 2:
                        row = {f"column_{i+1}": val for i, val in enumerate(values)}
                        structured_data.append(row)
            
            return {
                "success": True,
                "structured_data": structured_data[:50],  # Limit to 50 entries
                "columns": list(structured_data[0].keys()) if structured_data else [],
                "total_extracted": len(structured_data),
                "method": "regex_fallback"
            }
            
        except Exception as e:
            logger.error(f"Fallback data structuring failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _basic_filter(self, table_data: List[Dict], filter_criteria: str, filter_type: str, limit: int) -> Dict[str, Any]:
        """Basic filtering fallback"""
        try:
            filtered_data = []
            criteria_lower = filter_criteria.lower()
            
            for row in table_data:
                # Simple text search across all fields
                row_text = ' '.join(str(v).lower() for v in row.values())
                if criteria_lower in row_text:
                    filtered_data.append(row)
                
                if len(filtered_data) >= limit:
                    break
            
            return {
                "success": True,
                "filtered_data": filtered_data,
                "total_matches": len(filtered_data),
                "returned_count": len(filtered_data),
                "filter_criteria": filter_criteria,
                "explanation": f"Basic text search for '{filter_criteria}'",
                "method": "text_search_fallback"
            }
            
        except Exception as e:
            logger.error(f"Basic filtering failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _basic_merge_tables(self, tables: List[Any], match_columns: List[str], merge_strategy: str) -> Dict[str, Any]:
        """Basic table merging fallback"""
        try:
            if not tables or len(tables) < 2:
                return {"success": False, "error": "Need at least 2 tables to merge"}
            
            # Simple concatenation if no match columns specified
            if not match_columns:
                merged_data = []
                for table in tables:
                    if isinstance(table, list):
                        merged_data.extend(table)
                
                return {
                    "success": True,
                    "merged_table": merged_data,
                    "merge_strategy": "concatenation",
                    "total_rows": len(merged_data),
                    "method": "concatenation_fallback"
                }
            
            # Simple key-based merge on first matching column
            merge_key = match_columns[0]
            base_table = tables[0] if isinstance(tables[0], list) else [tables[0]]
            
            for additional_table in tables[1:]:
                if not isinstance(additional_table, list):
                    additional_table = [additional_table]
                
                # Create lookup dictionary
                lookup = {row.get(merge_key): row for row in additional_table if row.get(merge_key)}
                
                # Merge data
                for row in base_table:
                    key_value = row.get(merge_key)
                    if key_value in lookup:
                        row.update(lookup[key_value])
            
            return {
                "success": True,
                "merged_table": base_table,
                "merge_strategy": merge_strategy,
                "join_columns": [merge_key],
                "total_rows": len(base_table),
                "method": "simple_join_fallback"
            }
            
        except Exception as e:
            logger.error(f"Basic table merging failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _basic_analyze_table(self, table_data: List[Dict], analysis_type: str, focus_columns: List[str]) -> Dict[str, Any]:
        """Basic table analysis fallback"""
        try:
            if not table_data:
                return {"success": False, "error": "No data to analyze"}
            
            analysis = {
                "summary": f"Basic analysis of {len(table_data)} rows",
                "key_metrics": {
                    "total_rows": len(table_data),
                    "total_columns": len(table_data[0].keys()),
                    "column_names": list(table_data[0].keys())
                },
                "insights": [
                    f"Dataset contains {len(table_data)} records",
                    f"Data has {len(table_data[0].keys())} columns",
                    "Use AI analysis for deeper insights"
                ],
                "data_quality": {
                    "completeness": 0.8,  # Rough estimate
                    "consistency": 0.7,   # Rough estimate
                    "issues": ["Use AI analysis for detailed quality assessment"]
                },
                "recommendations": [
                    "Enable AI analysis for comprehensive insights",
                    "Check for missing values",
                    "Validate data types"
                ],
                "method": "basic_stats_fallback"
            }
            
            return {"success": True, **analysis}
            
        except Exception as e:
            logger.error(f"Basic table analysis failed: {e}")
            return {"success": False, "error": str(e)}
    
    # Additional helper methods for AI implementations
    
    def _apply_ai_suggestions(self, table_data: List[Dict], suggestions: Dict[str, Any]) -> List[Dict]:
        """Apply AI suggestions to enhance table data"""
        try:
            enhanced_data = table_data.copy()
            
            # Apply column mappings
            column_mappings = suggestions.get("column_mappings", {})
            if column_mappings:
                for row in enhanced_data:
                    for old_name, new_name in column_mappings.items():
                        if old_name in row:
                            row[new_name] = row.pop(old_name)
            
            # Apply formatting rules
            formatting_rules = suggestions.get("formatting_rules", {})
            for column, rule in formatting_rules.items():
                for row in enhanced_data:
                    if column in row:
                        value = row[column]
                        if rule == "uppercase":
                            row[column] = str(value).upper()
                        elif rule == "lowercase":
                            row[column] = str(value).lower()
                        elif rule == "title_case":
                            row[column] = str(value).title()
                        elif rule.startswith("round_"):
                            places = int(rule.split("_")[1])
                            try:
                                row[column] = round(float(value), places)
                            except (ValueError, TypeError):
                                pass
            
            return enhanced_data
            
        except Exception as e:
            logger.error(f"Applying AI suggestions failed: {e}")
            return table_data
    
    def _apply_ai_filter_logic(self, table_data: List[Dict], filter_logic: Dict[str, Any]) -> List[Dict]:
        """Apply AI-generated filter logic to table data"""
        try:
            filtered_data = []
            main_filter = filter_logic.get("filter_logic", {})
            
            column = main_filter.get("column")
            operation = main_filter.get("operation")
            value = main_filter.get("value")
            
            if not column or not operation:
                return table_data
            
            for row in table_data:
                row_value = row.get(column)
                if row_value is None:
                    continue
                
                match = False
                
                if operation == "equals":
                    match = str(row_value).lower() == str(value).lower()
                elif operation == "contains":
                    match = str(value).lower() in str(row_value).lower()
                elif operation == "greater_than":
                    try:
                        match = float(row_value) > float(value)
                    except (ValueError, TypeError):
                        match = False
                elif operation == "less_than":
                    try:
                        match = float(row_value) < float(value)
                    except (ValueError, TypeError):
                        match = False
                elif operation == "range":
                    # Assume value is "min,max"
                    try:
                        min_val, max_val = map(float, str(value).split(','))
                        match = min_val <= float(row_value) <= max_val
                    except (ValueError, TypeError):
                        match = False
                
                if match:
                    filtered_data.append(row)
            
            # Apply sorting if specified
            sort_logic = filter_logic.get("sort_logic", {})
            if sort_logic and "column" in sort_logic:
                sort_column = sort_logic["column"]
                reverse = sort_logic.get("direction", "asc") == "desc"
                
                try:
                    filtered_data.sort(
                        key=lambda x: x.get(sort_column, ''), 
                        reverse=reverse
                    )
                except Exception as e:
                    logger.warning(f"Sorting failed: {e}")
            
            return filtered_data
            
        except Exception as e:
            logger.error(f"Applying AI filter logic failed: {e}")
            return table_data
    
    def _execute_ai_merge(self, tables: List[Any], merge_plan: Dict[str, Any], resolve_conflicts: str) -> Dict[str, Any]:
        """Execute AI-recommended merge plan"""
        try:
            strategy = merge_plan.get("recommended_strategy", "outer")
            join_columns = merge_plan.get("join_columns", [])
            
            if not join_columns or len(tables) < 2:
                # Fallback to concatenation
                merged_data = []
                for table in tables:
                    if isinstance(table, list):
                        merged_data.extend(table)
                return {"data": merged_data, "statistics": {"strategy": "concatenation"}}
            
            # Start with first table
            result_table = tables[0] if isinstance(tables[0], list) else [tables[0]]
            
            for i, additional_table in enumerate(tables[1:], 1):
                if not isinstance(additional_table, list):
                    additional_table = [additional_table]
                
                # Merge based on join columns
                merged_table = []
                
                for base_row in result_table:
                    base_keys = tuple(base_row.get(col, '') for col in join_columns)
                    
                    # Find matching rows in additional table
                    matches = [
                        row for row in additional_table
                        if tuple(row.get(col, '') for col in join_columns) == base_keys
                    ]
                    
                    if matches:
                        # Merge with first match (resolve conflicts based on strategy)
                        merged_row = base_row.copy()
                        for match in matches:
                            for key, value in match.items():
                                if key not in merged_row or merged_row[key] == '':
                                    merged_row[key] = value
                                elif resolve_conflicts == "keep_last":
                                    merged_row[key] = value
                                elif resolve_conflicts == "combine" and key not in join_columns:
                                    existing = str(merged_row[key])
                                    new_val = str(value)
                                    if existing != new_val:
                                        merged_row[key] = f"{existing}; {new_val}"
                        
                        merged_table.append(merged_row)
                    elif strategy in ["outer", "left"]:
                        merged_table.append(base_row)
                
                # Add unmatched rows from additional table for outer joins
                if strategy == "outer":
                    result_keys = {tuple(row.get(col, '') for col in join_columns) for row in result_table}
                    for row in additional_table:
                        row_keys = tuple(row.get(col, '') for col in join_columns)
                        if row_keys not in result_keys:
                            merged_table.append(row)
                
                result_table = merged_table
            
            return {
                "data": result_table,
                "statistics": {
                    "strategy": strategy,
                    "join_columns": join_columns,
                    "tables_merged": len(tables),
                    "final_rows": len(result_table)
                }
            }
            
        except Exception as e:
            logger.error(f"AI merge execution failed: {e}")
            return {"data": [], "statistics": {"error": str(e)}}
    
    async def _export_in_format(self, table_data: List[Dict], export_format: str, include_headers: bool, custom_styling: Dict, filename: str) -> Dict[str, Any]:
        """Export table data in specified format"""
        try:
            if export_format == "csv":
                content = self._convert_to_csv(table_data)
            elif export_format == "json":
                content = json.dumps(table_data, indent=2)
            elif export_format == "html":
                content = self._convert_to_html(table_data, custom_styling.get("style", "professional"))
                # Add HTML wrapper
                content = f"""<!DOCTYPE html>
<html><head><title>{filename}</title>
<style>
.table-professional {{ border-collapse: collapse; width: 100%; }}
.table-professional th, .table-professional td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
.table-professional th {{ background-color: #f2f2f2; }}
</style></head><body>{content}</body></html>"""
            elif export_format == "markdown":
                content = self._convert_to_markdown(table_data)
            elif export_format == "excel":
                # For Excel, we'll provide CSV format with Excel-specific metadata
                content = self._convert_to_csv(table_data)
            else:
                content = self._format_as_table(table_data)
            
            return {
                "content": content,
                "metadata": {
                    "format": export_format,
                    "headers_included": include_headers,
                    "rows_exported": len(table_data),
                    "export_timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return {"content": "", "metadata": {"error": str(e)}}
    
    def _convert_to_excel_info(self, table_data: List[Dict]) -> str:
        """Provide Excel export information"""
        csv_content = self._convert_to_csv(table_data)
        return f"""Excel Export Ready:
Rows: {len(table_data)}
Columns: {len(table_data[0].keys()) if table_data else 0}

CSV Content:
{csv_content}

Note: Save this content as a .csv file and open with Excel."""
    
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
server = EnhancedTabulatorMCPServer()
app = server.app

if __name__ == "__main__":
    print("=" * 60)
    print("📊 Starting Enhanced Tabulator MCP Server")
    print("=" * 60)
    print(f"🌐 Port: 9102")
    print(f"⚡ Server Name: Enhanced Tabulator Server")
    print(f"🔧 Capabilities: Smart Tabulation, AI Filtering, Structure Analysis, Data Merging")
    print(f"🤖 Azure OpenAI: Intelligent data processing and insights")
    print(f"📈 Formats: JSON, CSV, HTML, Markdown")
    print(f"📊 MCP Protocol: 2025-06-18")
    print("=" * 60)
    
    uvicorn.run(app, host="127.0.0.1", port=9102, log_level="info")