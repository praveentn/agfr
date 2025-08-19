# agentic/agents/local/sql_executor_server.py
import json
import time
import uuid
import re
import sys
import os
import sqlite3
import logging
import asyncio
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from fastapi import FastAPI, Request, HTTPException, Header, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn

from agentic.core.llm_client import llm_client
from agentic.core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

class DatabaseManager:
    """Enhanced database manager with security and performance optimizations"""
    
    def __init__(self, db_path: str = "agentic.db"):
        self.db_path = db_path
        self.connection_pool = {}
        self.max_connections = 10
        self._ensure_database_exists()
        self._create_system_tables()
    
    def _ensure_database_exists(self):
        """Ensure database file exists and is accessible"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("SELECT 1").fetchone()
            conn.close()
            logger.info(f"Database connection verified: {self.db_path}")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def _create_system_tables(self):
        """Create system tables for storing execution history and metadata"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Execution history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS execution_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    execution_time REAL NOT NULL,
                    rows_affected INTEGER,
                    error_message TEXT,
                    user_id TEXT,
                    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    query_type TEXT,
                    query_hash TEXT
                )
            """)
            
            # Query performance table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS query_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_hash TEXT UNIQUE,
                    query_pattern TEXT,
                    avg_execution_time REAL,
                    execution_count INTEGER DEFAULT 1,
                    last_executed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # System metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            conn.close()
            logger.info("System tables created/verified successfully")
            
        except Exception as e:
            logger.error(f"System table creation failed: {e}")
    
    async def execute_raw_sql(self, query: str, params: Optional[tuple] = None, user_id: str = None) -> Dict[str, Any]:
        """Execute raw SQL query with comprehensive error handling and logging"""
        start_time = time.time()
        trace_id = str(uuid.uuid4())
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            cursor = conn.cursor()
            
            # Execute query
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            # Determine query type
            query_type = self._get_query_type(query)
            
            # Handle different query types
            if query_type in ["SELECT", "PRAGMA", "EXPLAIN"]:
                rows = cursor.fetchall()
                columns = [description[0] for description in cursor.description] if cursor.description else []
                data = [dict(row) for row in rows]
                
                result = {
                    "success": True,
                    "data": data,
                    "columns": columns,
                    "row_count": len(data),
                    "query_type": query_type,
                    "execution_time": round(time.time() - start_time, 3),
                    "trace_id": trace_id
                }
                
            else:  # INSERT, UPDATE, DELETE, CREATE, DROP, etc.
                conn.commit()
                affected_rows = cursor.rowcount
                
                result = {
                    "success": True,
                    "affected_rows": affected_rows,
                    "query_type": query_type,
                    "execution_time": round(time.time() - start_time, 3),
                    "trace_id": trace_id,
                    "message": f"{query_type} completed successfully"
                }
            
            # Log execution
            await self._log_execution(trace_id, query, True, time.time() - start_time, 
                                    result.get("affected_rows"), None, user_id, query_type)
            
            conn.close()
            return result
            
        except sqlite3.Error as e:
            error_msg = str(e)
            execution_time = time.time() - start_time
            
            # Log failed execution
            await self._log_execution(trace_id, query, False, execution_time, 
                                    None, error_msg, user_id, self._get_query_type(query))
            
            logger.error(f"SQL execution failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "query_type": self._get_query_type(query),
                "execution_time": round(execution_time, 3),
                "trace_id": trace_id
            }
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            execution_time = time.time() - start_time
            
            await self._log_execution(trace_id, query, False, execution_time, 
                                    None, error_msg, user_id, "UNKNOWN")
            
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "execution_time": round(execution_time, 3),
                "trace_id": trace_id
            }
    
    def _get_query_type(self, query: str) -> str:
        """Determine the type of SQL query"""
        query_upper = query.strip().upper()
        
        if query_upper.startswith("SELECT"):
            return "SELECT"
        elif query_upper.startswith("INSERT"):
            return "INSERT"
        elif query_upper.startswith("UPDATE"):
            return "UPDATE"
        elif query_upper.startswith("DELETE"):
            return "DELETE"
        elif query_upper.startswith("CREATE"):
            return "CREATE"
        elif query_upper.startswith("DROP"):
            return "DROP"
        elif query_upper.startswith("ALTER"):
            return "ALTER"
        elif query_upper.startswith("PRAGMA"):
            return "PRAGMA"
        elif query_upper.startswith("EXPLAIN"):
            return "EXPLAIN"
        else:
            return "OTHER"
    
    async def _log_execution(self, trace_id: str, query: str, success: bool, 
                           execution_time: float, rows_affected: Optional[int], 
                           error_message: Optional[str], user_id: Optional[str], 
                           query_type: str):
        """Log query execution to history table"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query_hash = hash(re.sub(r'\s+', ' ', query.strip()))
            
            cursor.execute("""
                INSERT INTO execution_history 
                (trace_id, query, success, execution_time, rows_affected, error_message, user_id, query_type, query_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (trace_id, query, success, execution_time, rows_affected, error_message, user_id, query_type, str(query_hash)))
            
            # Update performance tracking
            if success:
                cursor.execute("""
                    INSERT OR REPLACE INTO query_performance 
                    (query_hash, query_pattern, avg_execution_time, execution_count, last_executed)
                    VALUES (?, ?, 
                        COALESCE((SELECT avg_execution_time * execution_count + ? FROM query_performance WHERE query_hash = ?) / 
                                (SELECT execution_count + 1 FROM query_performance WHERE query_hash = ?), ?),
                        COALESCE((SELECT execution_count + 1 FROM query_performance WHERE query_hash = ?), 1),
                        CURRENT_TIMESTAMP)
                """, (str(query_hash), query_type, execution_time, str(query_hash), str(query_hash), execution_time, str(query_hash)))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to log execution: {e}")
    
    async def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics and performance metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Overall stats
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_executions,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_executions,
                    SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed_executions,
                    AVG(execution_time) as avg_execution_time,
                    MAX(execution_time) as max_execution_time,
                    MIN(execution_time) as min_execution_time
                FROM execution_history
            """)
            overall_stats = dict(cursor.fetchone())
            
            # Query type distribution
            cursor.execute("""
                SELECT query_type, COUNT(*) as count, AVG(execution_time) as avg_time
                FROM execution_history
                GROUP BY query_type
                ORDER BY count DESC
            """)
            query_type_stats = [dict(row) for row in cursor.fetchall()]
            
            # Recent executions
            cursor.execute("""
                SELECT trace_id, query, success, execution_time, executed_at, query_type
                FROM execution_history
                ORDER BY executed_at DESC
                LIMIT 10
            """)
            recent_executions = [dict(row) for row in cursor.fetchall()]
            
            # Performance trends
            cursor.execute("""
                SELECT query_pattern, avg_execution_time, execution_count
                FROM query_performance
                ORDER BY execution_count DESC
                LIMIT 10
            """)
            performance_trends = [dict(row) for row in cursor.fetchall()]
            
            conn.close()
            
            return {
                "overall_stats": overall_stats,
                "query_type_distribution": query_type_stats,
                "recent_executions": recent_executions,
                "performance_trends": performance_trends,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get execution stats: {e}")
            return {"error": str(e)}

class SQLExecutorMCPServer:
    """Enhanced SQL Executor MCP Server with admin capabilities and security"""
    
    def __init__(self):
        self.app = FastAPI(title="SQL Executor MCP Server")
        self.db_manager = DatabaseManager()
        self.sessions = {}
        self.tools = [
            {
                "name": "execute_query",
                "description": "Execute raw SQL queries with comprehensive error handling and security validation",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "SQL query to execute", "minLength": 1},
                        "page": {"type": "integer", "minimum": 1, "default": 1, "description": "Page number for pagination"},
                        "page_size": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 50, "description": "Number of rows per page"},
                        "confirm_dangerous": {"type": "boolean", "default": False, "description": "Confirm execution of potentially dangerous operations"},
                        "user_id": {"type": "string", "description": "User identifier for logging"},
                        "dry_run": {"type": "boolean", "default": False, "description": "Validate query without executing"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "analyze_schema",
                "description": "Analyze database schema and provide structural information",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "table_name": {"type": "string", "description": "Specific table to analyze (optional)"},
                        "include_data_samples": {"type": "boolean", "default": False, "description": "Include sample data"},
                        "include_statistics": {"type": "boolean", "default": True, "description": "Include table statistics"}
                    }
                }
            },
            {
                "name": "optimize_query",
                "description": "Analyze and suggest optimizations for SQL queries using AI",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "SQL query to optimize", "minLength": 1},
                        "explain_plan": {"type": "boolean", "default": True, "description": "Include execution plan analysis"},
                        "suggest_indexes": {"type": "boolean", "default": True, "description": "Suggest index optimizations"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "generate_query",
                "description": "Generate SQL queries from natural language descriptions using AI",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string", "description": "Natural language description of desired query", "minLength": 5},
                        "target_tables": {"type": "array", "items": {"type": "string"}, "description": "Specific tables to query (optional)"},
                        "query_type": {"type": "string", "enum": ["SELECT", "INSERT", "UPDATE", "DELETE"], "default": "SELECT", "description": "Type of query to generate"},
                        "include_explanation": {"type": "boolean", "default": True, "description": "Include explanation of generated query"}
                    },
                    "required": ["description"]
                }
            },
            {
                "name": "validate_query",
                "description": "Validate SQL query syntax and security without execution",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "SQL query to validate", "minLength": 1},
                        "strict_mode": {"type": "boolean", "default": True, "description": "Enable strict validation mode"},
                        "check_permissions": {"type": "boolean", "default": True, "description": "Check permission requirements"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "explain_query",
                "description": "Explain SQL query execution plan and performance characteristics",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "SQL query to explain", "minLength": 1},
                        "detailed_analysis": {"type": "boolean", "default": True, "description": "Include detailed performance analysis"},
                        "ai_insights": {"type": "boolean", "default": True, "description": "Include AI-powered insights"}
                    },
                    "required": ["query"]
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
                "name": "SQL Executor Server",
                "version": "2.0.0",
                "description": "Administrative SQL execution with security and performance monitoring",
                "capabilities": [
                    "raw_sql_execution", "schema_analysis", "query_optimization",
                    "query_generation", "query_validation", "performance_monitoring"
                ],
                "protocol": "mcp-2025-06-18",
                "database": "sqlite",
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
                if tool_name == "execute_query":
                    result = await self._execute_query(arguments)
                elif tool_name == "analyze_schema":
                    result = await self._analyze_schema(arguments)
                elif tool_name == "optimize_query":
                    result = await self._optimize_query(arguments)
                elif tool_name == "generate_query":
                    result = await self._generate_query(arguments)
                elif tool_name == "validate_query":
                    result = await self._validate_query(arguments)
                elif tool_name == "explain_query":
                    result = await self._explain_query(arguments)
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
                # Test database connection
                db_status = "healthy"
                try:
                    result = await self.db_manager.execute_raw_sql("SELECT 1 as test")
                    if not result.get("success"):
                        db_status = "error"
                except:
                    db_status = "error"
                
                # Test Azure OpenAI connection
                llm_status = "healthy" if llm_client and llm_client.client else "not_configured"
                
                return {
                    "status": "healthy",
                    "server": "SQL Executor Server",
                    "database": db_status,
                    "azure_openai": llm_status,
                    "timestamp": time.time(),
                    "capabilities": [
                        "execute_query", "analyze_schema", "optimize_query",
                        "generate_query", "validate_query", "explain_query"
                    ]
                }
                
            except Exception as e:
                return {
                    "status": "error",
                    "error": str(e),
                    "timestamp": time.time()
                }
        
        @self.app.get("/admin/stats")
        async def get_admin_stats():
            """Get administrative statistics"""
            try:
                stats = await self.db_manager.get_execution_stats()
                return {"success": True, "statistics": stats}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
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
    
    # Core SQL Tools Implementation
    
    async def _execute_query(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute SQL query with pagination and security checks"""
        try:
            query = arguments.get("query", "").strip()
            page = arguments.get("page", 1)
            page_size = arguments.get("page_size", 50)
            confirm_dangerous = arguments.get("confirm_dangerous", False)
            user_id = arguments.get("user_id")
            dry_run = arguments.get("dry_run", False)
            
            if not query:
                return {"success": False, "error": "SQL query is required"}
            
            # Security validation
            security_check = await self._validate_query_security(query, confirm_dangerous)
            if not security_check["safe"]:
                return {
                    "success": False, 
                    "error": "Query blocked by security policy",
                    "security_warnings": security_check["warnings"],
                    "requires_confirmation": True,
                    "dangerous_keywords": security_check.get("dangerous_keywords", [])
                }
            
            # Dry run - validate without executing
            if dry_run:
                validation = await self._validate_query({"query": query, "strict_mode": True})
                return {
                    "success": True,
                    "dry_run": True,
                    "validation": validation,
                    "estimated_cost": "Low",  # Simple estimation
                    "query_type": self._get_query_type(query)
                }
            
            # Execute query
            result = await self.db_manager.execute_raw_sql(query, user_id=user_id)
            
            if not result["success"]:
                return result
            
            # Apply pagination for SELECT queries
            if result.get("data") and isinstance(result["data"], list):
                total_rows = len(result["data"])
                start_idx = (page - 1) * page_size
                end_idx = start_idx + page_size
                
                paginated_data = result["data"][start_idx:end_idx]
                
                result.update({
                    "data": paginated_data,
                    "pagination": {
                        "page": page,
                        "page_size": page_size,
                        "total_rows": total_rows,
                        "total_pages": (total_rows + page_size - 1) // page_size,
                        "has_next": end_idx < total_rows,
                        "has_prev": page > 1
                    }
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _analyze_schema(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze database schema structure"""
        try:
            table_name = arguments.get("table_name")
            include_samples = arguments.get("include_data_samples", False)
            include_stats = arguments.get("include_statistics", True)
            
            schema_info = {"tables": [], "indexes": [], "analysis_timestamp": datetime.now().isoformat()}
            
            if table_name:
                # Analyze specific table
                table_info = await self._analyze_single_table(table_name, include_samples, include_stats)
                if table_info["success"]:
                    schema_info["tables"] = [table_info["table_info"]]
                else:
                    return table_info
            else:
                # Analyze all tables
                tables_result = await self.db_manager.execute_raw_sql(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
                
                if tables_result["success"]:
                    for table_row in tables_result["data"]:
                        table_name = table_row["name"]
                        table_info = await self._analyze_single_table(table_name, include_samples, include_stats)
                        if table_info["success"]:
                            schema_info["tables"].append(table_info["table_info"])
            
            # Get index information
            indexes_result = await self.db_manager.execute_raw_sql(
                "SELECT name, tbl_name, sql FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
            )
            
            if indexes_result["success"]:
                schema_info["indexes"] = indexes_result["data"]
            
            return {
                "success": True,
                "schema": schema_info,
                "total_tables": len(schema_info["tables"]),
                "total_indexes": len(schema_info["indexes"])
            }
            
        except Exception as e:
            logger.error(f"Schema analysis failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _analyze_single_table(self, table_name: str, include_samples: bool, include_stats: bool) -> Dict[str, Any]:
        """Analyze a single table structure and statistics"""
        try:
            table_info = {"name": table_name, "columns": [], "constraints": []}
            
            # Get column information
            columns_result = await self.db_manager.execute_raw_sql(f"PRAGMA table_info({table_name})")
            if columns_result["success"]:
                table_info["columns"] = columns_result["data"]
            
            # Get foreign keys
            fk_result = await self.db_manager.execute_raw_sql(f"PRAGMA foreign_key_list({table_name})")
            if fk_result["success"]:
                table_info["foreign_keys"] = fk_result["data"]
            
            # Get statistics if requested
            if include_stats:
                count_result = await self.db_manager.execute_raw_sql(f"SELECT COUNT(*) as row_count FROM {table_name}")
                if count_result["success"]:
                    table_info["row_count"] = count_result["data"][0]["row_count"]
                
                # Get column statistics for numeric columns
                column_stats = {}
                for col in table_info["columns"]:
                    if col["type"].upper() in ["INTEGER", "REAL", "NUMERIC"]:
                        stats_query = f"""
                        SELECT 
                            MIN({col['name']}) as min_val,
                            MAX({col['name']}) as max_val,
                            AVG({col['name']}) as avg_val,
                            COUNT(DISTINCT {col['name']}) as distinct_count
                        FROM {table_name}
                        """
                        stats_result = await self.db_manager.execute_raw_sql(stats_query)
                        if stats_result["success"] and stats_result["data"]:
                            column_stats[col['name']] = stats_result["data"][0]
                
                table_info["column_statistics"] = column_stats
            
            # Get sample data if requested
            if include_samples:
                sample_result = await self.db_manager.execute_raw_sql(f"SELECT * FROM {table_name} LIMIT 5")
                if sample_result["success"]:
                    table_info["sample_data"] = sample_result["data"]
            
            return {"success": True, "table_info": table_info}
            
        except Exception as e:
            logger.error(f"Single table analysis failed for {table_name}: {e}")
            return {"success": False, "error": str(e)}
    
    async def _optimize_query(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize SQL query using AI analysis"""
        try:
            query = arguments.get("query", "").strip()
            explain_plan = arguments.get("explain_plan", True)
            suggest_indexes = arguments.get("suggest_indexes", True)
            
            if not query:
                return {"success": False, "error": "Query is required for optimization"}
            
            # Get execution plan
            execution_plan = {}
            if explain_plan:
                plan_result = await self.db_manager.execute_raw_sql(f"EXPLAIN QUERY PLAN {query}")
                if plan_result["success"]:
                    execution_plan = plan_result["data"]
            
            # Use AI for optimization suggestions if available
            if llm_client and llm_client.client:
                optimization_result = await self._ai_optimize_query(query, execution_plan, suggest_indexes)
            else:
                optimization_result = await self._basic_optimize_query(query, execution_plan)
            
            return {
                "success": True,
                "original_query": query,
                "execution_plan": execution_plan,
                "optimizations": optimization_result,
                "optimization_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Query optimization failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _generate_query(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Generate SQL query from natural language description"""
        try:
            description = arguments.get("description", "").strip()
            target_tables = arguments.get("target_tables", [])
            query_type = arguments.get("query_type", "SELECT")
            include_explanation = arguments.get("include_explanation", True)
            
            if not description:
                return {"success": False, "error": "Description is required for query generation"}
            
            # Use AI for query generation if available
            if llm_client and llm_client.client:
                generation_result = await self._ai_generate_query(description, target_tables, query_type, include_explanation)
            else:
                generation_result = await self._basic_generate_query(description, query_type)
            
            return generation_result
            
        except Exception as e:
            logger.error(f"Query generation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _validate_query(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Validate SQL query syntax and security"""
        try:
            query = arguments.get("query", "").strip()
            strict_mode = arguments.get("strict_mode", True)
            
            if not query:
                return {"success": False, "error": "SQL query is required"}
            
            # Syntax validation
            syntax_validation = await self._validate_query_syntax(query)
            
            # Security validation
            security_validation = await self._validate_query_security(query, False)
            
            # LLM-based validation if available
            llm_validation = {}
            if llm_client and llm_client.client:
                llm_validation = await self._get_llm_validation(query, strict_mode)
            
            validation_result = {
                "syntax": syntax_validation,
                "security": security_validation,
                "llm_analysis": llm_validation,
                "overall_valid": syntax_validation["valid"] and security_validation["safe"],
                "strict_mode": strict_mode
            }
            
            return {
                "success": True,
                "query": query,
                "validation": validation_result,
                "recommendations": self._get_validation_recommendations(validation_result)
            }
            
        except Exception as e:
            logger.error(f"Query validation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _explain_query(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Explain SQL query execution plan and performance"""
        try:
            query = arguments.get("query", "").strip()
            detailed_analysis = arguments.get("detailed_analysis", True)
            ai_insights = arguments.get("ai_insights", True)
            
            if not query:
                return {"success": False, "error": "Query is required for explanation"}
            
            # Get execution plan
            plan_result = await self.db_manager.execute_raw_sql(f"EXPLAIN QUERY PLAN {query}")
            execution_plan = plan_result["data"] if plan_result["success"] else []
            
            # Get detailed execution info
            explain_result = await self.db_manager.execute_raw_sql(f"EXPLAIN {query}")
            execution_details = explain_result["data"] if explain_result["success"] else []
            
            # Use AI for insights if available
            ai_analysis = {}
            if ai_insights and llm_client and llm_client.client:
                ai_analysis = await self._get_ai_query_insights(query, execution_plan, execution_details)
            
            explanation = {
                "query": query,
                "execution_plan": execution_plan,
                "execution_details": execution_details if detailed_analysis else [],
                "ai_insights": ai_analysis,
                "query_type": self._get_query_type(query),
                "complexity_estimate": self._estimate_query_complexity(query, execution_plan),
                "explanation_timestamp": datetime.now().isoformat()
            }
            
            return {"success": True, "explanation": explanation}
            
        except Exception as e:
            logger.error(f"Query explanation failed: {e}")
            return {"success": False, "error": str(e)}
    
    # AI-Powered Helper Methods
    
    async def _ai_optimize_query(self, query: str, execution_plan: List[Dict], suggest_indexes: bool) -> Dict[str, Any]:
        """AI-powered query optimization"""
        try:
            index_instruction = " Include index recommendations." if suggest_indexes else ""
            plan_context = f"\n\nExecution plan: {json.dumps(execution_plan, indent=2)}" if execution_plan else ""
            
            prompt = f"""
            Analyze this SQL query for optimization opportunities:
            
            Query: {query}{plan_context}
            {index_instruction}
            
            Provide optimization suggestions including:
            1. Query rewrite opportunities
            2. Index recommendations (if requested)
            3. Performance improvements
            4. Best practices violations
            
            Return as JSON:
            {{
                "optimized_query": "improved version of the query",
                "optimizations": [
                    {{
                        "type": "rewrite/index/structure",
                        "description": "what was changed",
                        "impact": "high/medium/low",
                        "reason": "why this improves performance"
                    }}
                ],
                "index_recommendations": [
                    {{
                        "table": "table_name",
                        "columns": ["col1", "col2"],
                        "index_sql": "CREATE INDEX...",
                        "estimated_benefit": "performance improvement description"
                    }}
                ],
                "performance_score": 85,
                "complexity_reduction": "30%"
            }}
            """
            
            messages = [
                {"role": "system", "content": "You are an expert SQL optimization specialist. Analyze queries and suggest specific improvements."},
                {"role": "user", "content": prompt}
            ]
            
            response = await llm_client.generate(messages, temperature=0.2, max_tokens=1000)
            
            if response:
                try:
                    return json.loads(response.strip())
                except json.JSONDecodeError:
                    return {"error": "Failed to parse AI optimization response", "raw_response": response}
            else:
                return await self._basic_optimize_query(query, execution_plan)
                
        except Exception as e:
            logger.error(f"AI query optimization failed: {e}")
            return await self._basic_optimize_query(query, execution_plan)
    
    async def _ai_generate_query(self, description: str, target_tables: List[str], query_type: str, include_explanation: bool) -> Dict[str, Any]:
        """AI-powered query generation"""
        try:
            table_context = f" Focus on these tables: {', '.join(target_tables)}." if target_tables else ""
            explanation_request = " Include detailed explanation of the generated query." if include_explanation else ""
            
            # Get schema context
            schema_context = ""
            if target_tables:
                for table in target_tables[:3]:  # Limit to avoid token overflow
                    table_info = await self._analyze_single_table(table, False, False)
                    if table_info["success"]:
                        schema_context += f"\nTable {table}: {table_info['table_info']['columns']}"
            
            prompt = f"""
            Generate a {query_type} SQL query based on this description:
            
            Description: "{description}"
            {table_context}{explanation_request}
            
            Schema context: {schema_context}
            
            Generate a valid SQL query that accomplishes the requested task.
            
            Return as JSON:
            {{
                "generated_query": "SELECT * FROM ...",
                "query_type": "{query_type}",
                "explanation": "Step by step explanation of what the query does",
                "assumptions": ["assumption1", "assumption2"],
                "confidence": 0.9,
                "alternative_approaches": ["approach1", "approach2"]
            }}
            """
            
            messages = [
                {"role": "system", "content": "You are an expert SQL developer. Generate accurate, efficient SQL queries from natural language descriptions."},
                {"role": "user", "content": prompt}
            ]
            
            response = await llm_client.generate(messages, temperature=0.3, max_tokens=800)
            
            if response:
                try:
                    result = json.loads(response.strip())
                    return {"success": True, **result}
                except json.JSONDecodeError:
                    return {"success": False, "error": "Failed to parse AI generation response", "raw_response": response}
            else:
                return await self._basic_generate_query(description, query_type)
                
        except Exception as e:
            logger.error(f"AI query generation failed: {e}")
            return await self._basic_generate_query(description, query_type)
    
    async def _get_llm_validation(self, query: str, strict_mode: bool) -> Dict[str, Any]:
        """Get LLM-based query validation and suggestions"""
        try:
            strict_instruction = " Use strict validation criteria." if strict_mode else ""
            
            prompt = f"""
            Validate this SQL query for correctness, performance, and best practices:
            
            Query: {query}
            {strict_instruction}
            
            Analyze for:
            1. Syntax correctness
            2. Performance issues
            3. Security concerns
            4. Best practices compliance
            5. Potential improvements
            
            Return as JSON:
            {{
                "overall_score": 85,
                "issues": [
                    {{
                        "type": "performance/security/syntax/style",
                        "severity": "high/medium/low",
                        "description": "what the issue is",
                        "suggestion": "how to fix it"
                    }}
                ],
                "recommendations": ["rec1", "rec2"],
                "compliance_score": 90,
                "performance_score": 75
            }}
            """
            
            messages = [
                {"role": "system", "content": "You are an expert SQL validator. Analyze queries for correctness, performance, and best practices."},
                {"role": "user", "content": prompt}
            ]
            
            response = await llm_client.generate(messages, temperature=0.1, max_tokens=600)
            
            if response:
                try:
                    return json.loads(response.strip())
                except json.JSONDecodeError:
                    return {"error": "Failed to parse LLM validation response"}
            else:
                return {"error": "No LLM response received"}
                
        except Exception as e:
            logger.error(f"LLM validation failed: {e}")
            return {"error": str(e)}
    
    async def _get_ai_query_insights(self, query: str, execution_plan: List[Dict], execution_details: List[Dict]) -> Dict[str, Any]:
        """Get AI insights about query execution"""
        try:
            prompt = f"""
            Analyze this SQL query execution for insights and recommendations:
            
            Query: {query}
            Execution Plan: {json.dumps(execution_plan, indent=2)}
            
            Provide insights about:
            1. Execution efficiency
            2. Resource usage patterns
            3. Potential bottlenecks
            4. Scalability concerns
            
            Return as JSON:
            {{
                "efficiency_score": 85,
                "bottlenecks": ["bottleneck1", "bottleneck2"],
                "resource_usage": {{
                    "cpu_intensity": "low/medium/high",
                    "memory_usage": "estimate",
                    "io_operations": "read/write pattern analysis"
                }},
                "scalability_assessment": "how well this query will scale",
                "recommendations": ["rec1", "rec2"]
            }}
            """
            
            messages = [
                {"role": "system", "content": "You are an expert database performance analyst. Provide actionable insights about query execution."},
                {"role": "user", "content": prompt}
            ]
            
            response = await llm_client.generate(messages, temperature=0.2, max_tokens=600)
            
            if response:
                try:
                    return json.loads(response.strip())
                except json.JSONDecodeError:
                    return {"error": "Failed to parse AI insights response"}
            else:
                return {"error": "No AI response received"}
                
        except Exception as e:
            logger.error(f"AI query insights failed: {e}")
            return {"error": str(e)}
    
    # Fallback and Utility Methods
    
    async def _validate_query_syntax(self, query: str) -> Dict[str, Any]:
        """Validate SQL query syntax"""
        try:
            # Try to parse with SQLite (without executing)
            conn = sqlite3.connect(":memory:")
            cursor = conn.cursor()
            
            # Basic syntax check by attempting to prepare the statement
            try:
                cursor.execute("EXPLAIN QUERY PLAN " + query)
                return {"valid": True, "errors": []}
            except sqlite3.Error as e:
                return {"valid": False, "errors": [str(e)]}
            finally:
                conn.close()
                
        except Exception as e:
            return {"valid": False, "errors": [f"Syntax validation failed: {str(e)}"]}
    
    async def _validate_query_security(self, query: str, confirm_dangerous: bool) -> Dict[str, Any]:
        """Validate query for security risks"""
        try:
            query_upper = query.upper().strip()
            warnings = []
            
            # Check for dangerous operations
            dangerous_keywords = ["DROP", "DELETE", "TRUNCATE", "ALTER"]
            risky_keywords = ["CREATE", "INSERT", "UPDATE"]
            
            dangerous_found = [kw for kw in dangerous_keywords if kw in query_upper]
            risky_found = [kw for kw in risky_keywords if kw in query_upper]
            
            if dangerous_found:
                warnings.append(f"Dangerous operations detected: {', '.join(dangerous_found)}")
            
            if risky_found:
                warnings.append(f"Data modification operations: {', '.join(risky_found)}")
            
            # Check for potential SQL injection patterns
            injection_patterns = [
                r";\s*(DROP|DELETE|ALTER)",
                r"UNION\s+SELECT",
                r"--\s*\w+",
                r"/\*.*\*/"
            ]
            
            for pattern in injection_patterns:
                if re.search(pattern, query_upper):
                    warnings.append("Potential SQL injection pattern detected")
                    break
            
            # Determine if query is safe
            is_safe = True
            if dangerous_found and not confirm_dangerous:
                is_safe = False
            
            return {
                "safe": is_safe,
                "warnings": warnings,
                "dangerous_keywords": dangerous_found,
                "risky_keywords": risky_found,
                "requires_confirmation": bool(dangerous_found or risky_found)
            }
            
        except Exception as e:
            return {"safe": False, "warnings": [f"Security validation failed: {str(e)}"]}
    
    async def _basic_optimize_query(self, query: str, execution_plan: List[Dict]) -> Dict[str, Any]:
        """Basic query optimization fallback"""
        try:
            optimizations = []
            
            # Simple pattern-based suggestions
            query_upper = query.upper()
            
            if "SELECT *" in query_upper:
                optimizations.append({
                    "type": "rewrite",
                    "description": "Replace SELECT * with specific columns",
                    "impact": "medium",
                    "reason": "Reduces data transfer and improves performance"
                })
            
            if "ORDER BY" in query_upper and "LIMIT" not in query_upper:
                optimizations.append({
                    "type": "structure",
                    "description": "Consider adding LIMIT clause with ORDER BY",
                    "impact": "medium",
                    "reason": "Prevents unnecessary sorting of large result sets"
                })
            
            if not optimizations:
                optimizations.append({
                    "type": "analysis",
                    "description": "No obvious optimization opportunities detected",
                    "impact": "low",
                    "reason": "Query appears to follow basic best practices"
                })
            
            return {
                "optimizations": optimizations,
                "performance_score": 75,  # Default score
                "method": "basic_pattern_analysis"
            }
            
        except Exception as e:
            logger.error(f"Basic query optimization failed: {e}")
            return {"error": str(e)}
    
    async def _basic_generate_query(self, description: str, query_type: str) -> Dict[str, Any]:
        """Basic query generation fallback"""
        try:
            # Simple template-based generation
            if "all" in description.lower() and "from" in description.lower():
                # Try to extract table name
                words = description.lower().split()
                table_name = "your_table"
                if "from" in words:
                    from_idx = words.index("from")
                    if from_idx + 1 < len(words):
                        table_name = words[from_idx + 1].rstrip(".,!?")
                
                generated_query = f"SELECT * FROM {table_name}"
            elif query_type == "SELECT":
                generated_query = "SELECT column1, column2 FROM your_table WHERE condition = 'value'"
            elif query_type == "INSERT":
                generated_query = "INSERT INTO your_table (column1, column2) VALUES ('value1', 'value2')"
            elif query_type == "UPDATE":
                generated_query = "UPDATE your_table SET column1 = 'new_value' WHERE condition = 'value'"
            elif query_type == "DELETE":
                generated_query = "DELETE FROM your_table WHERE condition = 'value'"
            else:
                generated_query = "-- Unable to generate query from description"
            
            return {
                "success": True,
                "generated_query": generated_query,
                "query_type": query_type,
                "explanation": "Basic template-based query generation",
                "confidence": 0.5,
                "method": "template_fallback",
                "note": "This is a basic template. Please customize with your actual table and column names."
            }
            
        except Exception as e:
            logger.error(f"Basic query generation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _get_query_type(self, query: str) -> str:
        """Get the type of SQL query"""
        return self.db_manager._get_query_type(query)
    
    def _estimate_query_complexity(self, query: str, execution_plan: List[Dict]) -> str:
        """Estimate query complexity"""
        try:
            query_upper = query.upper()
            complexity_factors = 0
            
            # Count complexity indicators
            if "JOIN" in query_upper:
                complexity_factors += query_upper.count("JOIN")
            if "SUBQUERY" in query_upper or "(" in query:
                complexity_factors += 1
            if "GROUP BY" in query_upper:
                complexity_factors += 1
            if "ORDER BY" in query_upper:
                complexity_factors += 1
            if "HAVING" in query_upper:
                complexity_factors += 1
            
            # Estimate based on execution plan
            if execution_plan:
                plan_steps = len(execution_plan)
                if plan_steps > 5:
                    complexity_factors += 2
                elif plan_steps > 3:
                    complexity_factors += 1
            
            if complexity_factors >= 4:
                return "high"
            elif complexity_factors >= 2:
                return "medium"
            else:
                return "low"
                
        except Exception:
            return "unknown"
    
    def _get_validation_recommendations(self, validation_result: Dict[str, Any]) -> List[str]:
        """Get recommendations based on validation results"""
        recommendations = []
        
        if not validation_result["syntax"]["valid"]:
            recommendations.append("Fix syntax errors before execution")
        
        if not validation_result["security"]["safe"]:
            recommendations.append("Review security warnings and confirm dangerous operations")
        
        if validation_result["security"]["requires_confirmation"]:
            recommendations.append("Use confirm_dangerous=true to execute data modification queries")
        
        llm_analysis = validation_result.get("llm_analysis", {})
        if "recommendations" in llm_analysis:
            recommendations.extend(llm_analysis["recommendations"])
        
        if not recommendations:
            recommendations.append("Query validation passed successfully")
        
        return recommendations
    
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
server = SQLExecutorMCPServer()
app = server.app

if __name__ == "__main__":
    print("=" * 60)
    print("💾 Starting SQL Executor MCP Server")
    print("=" * 60)
    print(f"🌐 Port: 9104")
    print(f"⚡ Server Name: SQL Executor Server")
    print(f"🔧 Capabilities: Raw SQL Execution, Schema Analysis, Query Optimization")
    print(f"🤖 Azure OpenAI: Intelligent query assistance and optimization")
    print(f"🛡️ Security: Query validation and dangerous operation protection")
    print(f"📊 MCP Protocol: 2025-06-18")
    print("=" * 60)
    
    uvicorn.run(app, host="127.0.0.1", port=9104, log_level="info")