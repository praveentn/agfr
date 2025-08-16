# ============================================================================
# agentic/app/main.py
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import asyncio
from pathlib import Path
import os

from ..core.config import settings
from ..core.types import QueryRequest, QueryResponse, ExecutionPlan
from ..core.registry import registry
from ..core.planner import planner
from ..core.orchestrator import orchestrator
from ..core.composer import composer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
    description="Multi-agent workflow orchestration system",
    version="1.0.0"
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
        # Create execution plan
        dag = await planner.create_plan(request)
        trace_id = f"trace_{int(time.time() * 1000)}"
        
        # Execute workflow
        results = await orchestrator.execute_dag(dag, trace_id)
        
        # Compose final result
        final_result = composer.compose_results(results, dag.metadata.get("intent", "general"))
        
        return QueryResponse(
            trace_id=trace_id,
            intent=dag.metadata.get("intent", "unknown"),
            plan=dag,
            results=results,
            final_result=final_result,
            execution_time=sum(r.finished_at - r.started_at for r in results),
            success=all(r.success for r in results),
            error=None if all(r.success for r in results) else "Some steps failed"
        )
        
    except Exception as e:
        logger.error(f"Query execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/agents")
async def list_agents(token: str = Depends(verify_token)):
    """List all available agents and their tools"""
    agents = registry.list_agents()
    return {
        "agents": [
            {
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
            for agent in agents
        ]
    }

@app.get("/api/workflows")
async def list_workflows(token: str = Depends(verify_token)):
    """List all available workflows"""
    workflows = registry.list_workflows()
    return {
        "workflows": [
            {
                "id": wf.id,
                "name": wf.name,
                "description": wf.description,
                "intent": wf.intent,
                "nodes": len(wf.plan.nodes)
            }
            for wf in workflows
        ]
    }

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
            "results": results,
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
    """Execute raw SQL query (admin functionality)"""
    try:
        import sqlite3
        
        sql_query = query.get("query", "").strip()
        if not sql_query:
            raise HTTPException(status_code=400, detail="SQL query is required")
        
        # Basic SQL injection protection
        dangerous_keywords = ["DROP", "DELETE", "TRUNCATE", "ALTER"]
        if any(keyword in sql_query.upper() for keyword in dangerous_keywords):
            if not query.get("confirm_dangerous", False):
                raise HTTPException(
                    status_code=400, 
                    detail="Dangerous SQL detected. Use confirm_dangerous=true to proceed."
                )
        
        db_path = settings.database_url.replace("sqlite:///", "")
        
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute(sql_query)
            
            if sql_query.upper().strip().startswith("SELECT"):
                rows = cursor.fetchall()
                columns = [description[0] for description in cursor.description] if cursor.description else []
                data = [dict(row) for row in rows]
                
                return {
                    "success": True,
                    "data": data,
                    "columns": columns,
                    "row_count": len(data)
                }
            else:
                conn.commit()
                return {
                    "success": True,
                    "message": f"Query executed successfully. Rows affected: {cursor.rowcount}",
                    "rows_affected": cursor.rowcount
                }
                
    except Exception as e:
        logger.error(f"SQL execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/registry/reload")
async def reload_registry(token: str = Depends(verify_token)):
    """Reload registry from filesystem"""
    try:
        registry.reload()
        return {"success": True, "message": "Registry reloaded successfully"}
    except Exception as e:
        logger.error(f"Registry reload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")

