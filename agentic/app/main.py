# agentic/app/main.py
from fastapi import FastAPI, HTTPException, Depends, Security, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import asyncio
import time
from pathlib import Path
import os
import sqlite3
import json
from typing import List, Dict, Any, Optional

from ..core.config import settings
from ..core.types import QueryRequest, QueryResponse, ExecutionPlan
from ..core.registry import registry
from ..core.planner import planner
from ..core.orchestrator import orchestrator
from ..core.composer import composer
from ..core.database import db_manager
from ..core.mcp_client import MCPClientManager

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format=settings.log_format
)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != settings.auth_token:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return credentials.credentials

# Create FastAPI app
app = FastAPI(
    title="Agentic Framework",
    description="Multi-agent workflow orchestration system with MCP protocol support",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Create global MCP client manager
mcp_client_manager = MCPClientManager()

# API Routes
@app.get("/")
async def root():
    """Serve the main web interface"""
    return FileResponse(Path(__file__).parent / "static" / "index.html")

@app.post("/api/query", response_model=QueryResponse)
async def execute_query(
    request: QueryRequest,
    token: str = Depends(verify_token)
) -> QueryResponse:
    """Execute user query with workflow orchestration"""
    try:
        start_time = time.time()
        
        # Create execution plan
        dag = await planner.create_plan(request)
        trace_id = f"trace_{int(time.time() * 1000)}"
        
        logger.info(f"Executing query: {request.text} with trace_id: {trace_id}")
        
        # Execute workflow
        results = await orchestrator.execute_dag(dag, trace_id)
        
        # Compose final result
        final_result = composer.compose_results(results, dag.metadata.get("intent", "general"))
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Log execution to database
        try:
            for result in results:
                db_manager.execute_query("""
                    INSERT INTO execution_logs 
                    (trace_id, node_id, agent, tool, success, started_at, finished_at, data, error)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trace_id, result.node_id, result.agent, result.tool, result.success,
                    result.started_at, result.finished_at, 
                    str(result.data)[:1000] if result.data else None,  # Truncate large data
                    result.error
                ))
            
            # Log workflow execution
            db_manager.execute_query("""
                INSERT INTO workflow_executions 
                (trace_id, intent, query, status, final_result, execution_time, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                trace_id, dag.metadata.get("intent", "unknown"), request.text,
                "completed" if all(r.success for r in results) else "partial_failure",
                str(final_result)[:2000], round(execution_time, 3), time.time()
            ))
        except Exception as e:
            logger.error(f"Failed to log execution: {e}")
        
        return QueryResponse(
            trace_id=trace_id,
            intent=dag.metadata.get("intent", "unknown"),
            plan=dag,
            results=results,
            final_result=final_result,
            execution_time=round(execution_time, 3),
            success=all(r.success for r in results),
            error=None if all(r.success for r in results) else "Some steps failed"
        )
        
    except Exception as e:
        logger.error(f"Query execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/agents")
async def list_agents(
    include_health: bool = Query(False, description="Include health status for each agent"),
    token: str = Depends(verify_token)
):
    """List all available agents and their tools with optional health checks"""
    try:
        agents = registry.list_agents()
        result_agents = []
        
        for agent in agents:
            agent_info = {
                "name": agent.name,
                "description": agent.description,
                "endpoint": agent.endpoint,
                "enabled": agent.enabled,
                "tools": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "params_schema": tool.params_schema
                    }
                    for tool in agent.tools
                ]
            }
            
            if include_health:
                try:
                    # Test agent health
                    health_status = await mcp_client_manager.test_agent_connection(agent)
                    agent_info["health"] = {
                        "status": "healthy" if health_status else "unhealthy",
                        "last_checked": time.time()
                    }
                    
                    # Get live tools if agent is healthy
                    if health_status:
                        try:
                            live_tools = await mcp_client_manager.get_agent_tools(agent)
                            if live_tools:
                                agent_info["live_tools"] = live_tools
                        except Exception as e:
                            logger.debug(f"Could not get live tools for {agent.name}: {e}")
                            
                except Exception as e:
                    logger.error(f"Health check failed for {agent.name}: {e}")
                    agent_info["health"] = {
                        "status": "error",
                        "error": str(e),
                        "last_checked": time.time()
                    }
            
            result_agents.append(agent_info)
        
        return {
            "agents": result_agents,
            "total_agents": len(result_agents),
            "enabled_agents": len([a for a in agents if a.enabled])
        }
        
    except Exception as e:
        logger.error(f"Failed to list agents: {e}")
        return {"agents": [], "error": str(e)}

@app.get("/api/agents/{agent_name}/tools")
async def get_agent_tools(
    agent_name: str,
    token: str = Depends(verify_token)
):
    """Get detailed information about an agent's tools"""
    try:
        agent = registry.get_agent(agent_name)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
        
        # Get live tools from the agent if possible
        try:
            live_tools = await mcp_client_manager.get_agent_tools(agent)
            return {
                "agent_name": agent_name,
                "tools": live_tools,
                "source": "live",
                "total_tools": len(live_tools)
            }
        except Exception as e:
            logger.warning(f"Could not get live tools for {agent_name}: {e}")
            
            # Fallback to registry tools
            return {
                "agent_name": agent_name,
                "tools": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.params_schema
                    }
                    for tool in agent.tools
                ],
                "source": "registry",
                "total_tools": len(agent.tools),
                "note": "Live tools unavailable, showing registry tools"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get tools for {agent_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/agents/{agent_name}/tools/{tool_name}")
async def test_agent_tool(
    agent_name: str,
    tool_name: str,
    params: Dict[str, Any],
    token: str = Depends(verify_token)
):
    """Test a specific tool on an agent"""
    try:
        agent = registry.get_agent(agent_name)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
        
        # Call the tool
        result = await mcp_client_manager.call_tool(agent, tool_name, params)
        
        return {
            "agent_name": agent_name,
            "tool_name": tool_name,
            "params": params,
            "result": result,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to test tool {agent_name}.{tool_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/agents/{agent_name}/health")
async def check_agent_health(
    agent_name: str,
    token: str = Depends(verify_token)
):
    """Check health status of a specific agent"""
    try:
        agent = registry.get_agent(agent_name)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
        
        health_status = await mcp_client_manager.test_agent_connection(agent)
        
        health_info = {
            "agent_name": agent_name,
            "status": "healthy" if health_status else "unhealthy",
            "endpoint": agent.endpoint,
            "last_checked": time.time()
        }
        
        # Try to get additional health info from the agent
        if health_status:
            try:
                health_result = await mcp_client_manager.call_tool(agent, "health", {}, timeout=5)
                if health_result.get("success"):
                    health_info["details"] = health_result.get("data", {})
            except Exception as e:
                logger.debug(f"Could not get detailed health for {agent_name}: {e}")
        
        return health_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check failed for {agent_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/workflows")
async def list_workflows(token: str = Depends(verify_token)):
    """List all available workflows"""
    try:
        workflows = registry.list_workflows()
        return {
            "workflows": [
                {
                    "id": wf.id,
                    "name": wf.name,
                    "description": wf.description,
                    "intent": wf.intent,
                    "nodes": len(wf.plan.nodes),
                    "metadata": wf.metadata
                }
                for wf in workflows
            ],
            "total_workflows": len(workflows)
        }
    except Exception as e:
        logger.error(f"Failed to list workflows: {e}")
        return {"workflows": [], "error": str(e)}

@app.post("/api/workflows/{workflow_id}/run")
async def run_workflow(
    workflow_id: str,
    inputs: dict = {},
    token: str = Depends(verify_token)
):
    """Run a specific workflow by ID"""
    workflow = registry.get_workflow(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    try:
        trace_id = f"workflow_{workflow_id}_{int(time.time() * 1000)}"
        results = await orchestrator.execute_dag(workflow.plan, trace_id)
        final_result = composer.compose_results(results, workflow.intent)
        
        return {
            "trace_id": trace_id,
            "workflow_id": workflow_id,
            "results": [r.dict() for r in results],
            "final_result": final_result,
            "success": all(r.success for r in results)
        }
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/admin/sql")
async def execute_sql(
    query: dict,
    token: str = Depends(verify_token)
):
    """Execute raw SQL query (admin functionality) with enhanced security and pagination"""
    try:
        sql_query = query.get("query", "").strip()
        if not sql_query:
            raise HTTPException(status_code=400, detail="SQL query is required")
        
        # Basic SQL injection protection and validation
        dangerous_keywords = ["DROP", "DELETE", "TRUNCATE", "ALTER", "CREATE INDEX", "DROP INDEX"]
        sql_upper = sql_query.upper()
        
        is_dangerous = any(keyword in sql_upper for keyword in dangerous_keywords)
        is_confirm_dangerous = query.get("confirm_dangerous", False)
        
        if is_dangerous and not is_confirm_dangerous:
            return {
                "success": False,
                "error": "Potentially dangerous SQL detected",
                "dangerous_keywords": [kw for kw in dangerous_keywords if kw in sql_upper],
                "suggestion": "Use confirm_dangerous=true to proceed with dangerous operations",
                "query_preview": sql_query[:100] + "..." if len(sql_query) > 100 else sql_query,
                "requires_confirmation": True
            }
        
        # Pagination for SELECT queries
        page = max(1, query.get("page", 1))
        page_size = min(max(1, query.get("page_size", 50)), 1000)  # Between 1-1000 records per page
        
        # Add LIMIT clause for SELECT queries if not present
        original_query = sql_query
        if sql_upper.startswith("SELECT") and "LIMIT" not in sql_upper:
            offset = (page - 1) * page_size
            sql_query += f" LIMIT {page_size} OFFSET {offset}"
        
        # Execute the query
        start_time = time.time()
        result = db_manager.execute_query(sql_query)
        execution_time = round((time.time() - start_time) * 1000, 2)  # Convert to milliseconds
        
        if result.get("success"):
            # Enhance result with pagination info
            if "data" in result:
                # Get total count for SELECT queries with pagination
                total_rows = None
                if sql_upper.startswith("SELECT") and "LIMIT" in sql_query and page == 1:
                    try:
                        # Try to get total count
                        count_query = f"SELECT COUNT(*) as total FROM ({original_query}) as count_subquery"
                        count_result = db_manager.execute_query(count_query)
                        if count_result.get("success") and count_result.get("data"):
                            total_rows = count_result["data"][0]["total"]
                    except Exception as e:
                        logger.debug(f"Could not get total count: {e}")
                
                result["pagination"] = {
                    "page": page,
                    "page_size": page_size,
                    "execution_time_ms": execution_time,
                    "total_rows": total_rows,
                    "has_more": len(result["data"]) == page_size if result["data"] else False
                }
                
                # Add query analysis
                result["query_analysis"] = {
                    "type": _analyze_query_type(sql_query),
                    "is_dangerous": is_dangerous,
                    "estimated_complexity": _estimate_query_complexity(sql_query),
                    "original_query": original_query if original_query != sql_query else None
                }
        
        # Round decimal values in data
        if result.get("data"):
            result["data"] = _round_decimal_values(result["data"])
        
        # Log the query execution
        try:
            log_entry = {
                "query": sql_query[:500],  # Truncate long queries
                "success": result.get("success"),
                "execution_time_ms": execution_time,
                "row_count": result.get("row_count", 0),
                "error": result.get("error"),
                "timestamp": time.time(),
                "is_dangerous": is_dangerous
            }
            logger.info(f"SQL query executed: {log_entry}")
        except Exception as e:
            logger.warning(f"Failed to log SQL execution: {e}")
        
        return result
                
    except Exception as e:
        logger.error(f"SQL execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def _round_decimal_values(data):
    """Round decimal values in database results to 2-3 decimal places"""
    if isinstance(data, list):
        return [_round_decimal_values(item) for item in data]
    elif isinstance(data, dict):
        return {key: _round_decimal_values(value) for key, value in data.items()}
    elif isinstance(data, float):
        # Round to 3 decimal places for better precision
        return round(data, 3)
    else:
        return data

def _analyze_query_type(query: str) -> str:
    """Analyze SQL query type"""
    query_upper = query.upper().strip()
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
    else:
        return "OTHER"

def _estimate_query_complexity(query: str) -> str:
    """Estimate query complexity based on keywords"""
    complexity_indicators = {
        "JOIN": 1,
        "SUBQUERY": 2,
        "GROUP BY": 1,
        "ORDER BY": 1,
        "HAVING": 1,
        "UNION": 2,
        "WITH": 2,
        "CASE": 1
    }
    
    query_upper = query.upper()
    total_complexity = 0
    
    for indicator, weight in complexity_indicators.items():
        total_complexity += query_upper.count(indicator) * weight
    
    if total_complexity == 0:
        return "SIMPLE"
    elif total_complexity <= 3:
        return "MODERATE"
    else:
        return "COMPLEX"

@app.get("/api/admin/tables")
async def list_database_tables(token: str = Depends(verify_token)):
    """List all tables in the database with enhanced metadata"""
    try:
        result = db_manager.execute_query("""
            SELECT name, type FROM sqlite_master 
            WHERE type IN ('table', 'view') 
            ORDER BY type, name
        """)
        
        if result.get("success"):
            tables_info = []
            for table in result.get("data", []):
                table_name = table["name"]
                
                # Get table info
                try:
                    table_info_result = db_manager.execute_query(f"PRAGMA table_info({table_name})")
                    columns = table_info_result.get("data", []) if table_info_result.get("success") else []
                    
                    # Get row count
                    count_result = db_manager.execute_query(f"SELECT COUNT(*) as count FROM {table_name}")
                    row_count = count_result.get("data", [{}])[0].get("count", 0) if count_result.get("success") else 0
                    
                    # Get sample data
                    sample_result = db_manager.execute_query(f"SELECT * FROM {table_name} LIMIT 3")
                    sample_data = sample_result.get("data", []) if sample_result.get("success") else []
                    
                    tables_info.append({
                        "name": table_name,
                        "type": table["type"],
                        "columns": len(columns),
                        "column_details": columns,
                        "row_count": row_count,
                        "sample_data": sample_data,
                        "quick_queries": [
                            f"SELECT * FROM {table_name} LIMIT 10",
                            f"SELECT COUNT(*) FROM {table_name}",
                            f"SELECT * FROM {table_name} ORDER BY rowid DESC LIMIT 5"
                        ]
                    })
                except Exception as e:
                    tables_info.append({
                        "name": table_name,
                        "type": table["type"],
                        "error": f"Could not get table info: {str(e)}"
                    })
            
            return {
                "success": True,
                "tables": tables_info,
                "total_tables": len(tables_info),
                "database_size": _get_database_size()
            }
        else:
            return result
            
    except Exception as e:
        logger.error(f"Failed to list tables: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def _get_database_size():
    """Get database file size"""
    try:
        db_path = settings.database_url.replace("sqlite:///", "")
        if os.path.exists(db_path):
            size_bytes = os.path.getsize(db_path)
            size_mb = round(size_bytes / (1024 * 1024), 2)
            return f"{size_mb} MB"
        return "Unknown"
    except:
        return "Unknown"

@app.post("/api/admin/sql/export")
async def export_sql_results(
    query: dict,
    token: str = Depends(verify_token)
):
    """Export SQL query results to CSV format"""
    try:
        sql_query = query.get("query", "").strip()
        if not sql_query:
            raise HTTPException(status_code=400, detail="SQL query is required")
        
        # Execute the query (without pagination for export)
        result = db_manager.execute_query(sql_query)
        
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Query failed"))
        
        data = result.get("data", [])
        if not data:
            return {"success": True, "csv_data": "", "row_count": 0}
        
        # Convert to CSV
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=result.get("columns", data[0].keys()))
        writer.writeheader()
        writer.writerows(data)
        
        csv_content = output.getvalue()
        output.close()
        
        return {
            "success": True,
            "csv_data": csv_content,
            "row_count": len(data),
            "columns": result.get("columns", []),
            "filename": f"query_export_{int(time.time())}.csv"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"SQL export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/registry/reload")
async def reload_registry(token: str = Depends(verify_token)):
    """Reload registry from filesystem"""
    try:
        registry.reload()
        
        # Clear MCP client caches
        await mcp_client_manager.close_all()
        
        return {"success": True, "message": "Registry reloaded successfully"}
    except Exception as e:
        logger.error(f"Registry reload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/system/status")
async def system_status(token: str = Depends(verify_token)):
    """Get overall system status"""
    try:
        agents = registry.list_agents()
        workflows = registry.list_workflows()
        
        # Quick health check of all agents
        agent_health = {}
        for agent in agents:
            try:
                health = await mcp_client_manager.test_agent_connection(agent)
                agent_health[agent.name] = "healthy" if health else "unhealthy"
            except:
                agent_health[agent.name] = "error"
        
        healthy_agents = len([h for h in agent_health.values() if h == "healthy"])
        
        return {
            "system": {
                "status": "operational" if healthy_agents > 0 else "degraded",
                "uptime": time.time(),  # Simplified uptime
                "version": "1.0.0"
            },
            "agents": {
                "total": len(agents),
                "enabled": len([a for a in agents if a.enabled]),
                "healthy": healthy_agents,
                "unhealthy": len(agents) - healthy_agents,
                "details": agent_health
            },
            "workflows": {
                "total": len(workflows),
                "available": len([wf for wf in workflows if wf.intent])
            },
            "database": {
                "status": "connected",
                "url": settings.database_url,
                "size": _get_database_size()
            }
        }
        
    except Exception as e:
        logger.error(f"System status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0",
        "service": "Agentic Framework"
    }

# Cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    await mcp_client_manager.close_all()
    logger.info("Application shutdown complete")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host=settings.host, 
        port=settings.port, 
        log_level="info"
    )

