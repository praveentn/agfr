# agentic/app/main.py
from fastapi import FastAPI, HTTPException, Depends, Security, Query, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.websockets import WebSocket, WebSocketDisconnect
import logging
import asyncio
import time
from pathlib import Path
import os
import json
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..core.config import settings
from ..core.types import QueryRequest, QueryResponse, ExecutionPlan, WorkflowSpec
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
    description="Multi-agent workflow orchestration system with advanced workflow management and MCP protocol support",
    version="2.0.0",
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

# WebSocket connection manager for real-time updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except:
            self.disconnect(websocket)
    
    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                disconnected.append(connection)
        
        # Remove disconnected connections
        for conn in disconnected:
            self.disconnect(conn)

connection_manager = ConnectionManager()

# Background task for cleanup and maintenance
async def background_maintenance():
    """Background task for system maintenance"""
    while True:
        try:
            # Cleanup old execution data (older than 30 days)
            cleanup_stats = await db_manager.cleanup_old_data(30)
            logger.info(f"Background cleanup completed: {cleanup_stats}")
            
            # Wait 24 hours before next cleanup
            await asyncio.sleep(24 * 60 * 60)
        except Exception as e:
            logger.error(f"Background maintenance failed: {e}")
            await asyncio.sleep(60 * 60)  # Wait 1 hour before retry

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    try:
        # Initialize database
        await db_manager.initialize()
        
        # Initialize MCP client manager
        # await mcp_client_manager.initialize()
        # await mcp_client_manager._initialize_agent_session()

        # Start background maintenance task
        asyncio.create_task(background_maintenance())
        
        logger.info("Enhanced Agentic Framework started successfully")
    except Exception as e:
        logger.error(f"Startup failed: {e}")

# API Routes

@app.get("/")
async def root():
    """Serve the main web interface"""
    return FileResponse(Path(__file__).parent / "static" / "index.html")

@app.post("/api/query", response_model=QueryResponse)
async def execute_query(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """Execute a user query with enhanced workflow capabilities"""
    try:
        trace_id = f"query_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        
        logger.info(f"Executing query [{trace_id}]: {request.query}")
        
        # Broadcast execution start
        await connection_manager.broadcast(json.dumps({
            "type": "execution_start",
            "trace_id": trace_id,
            "query": request.query,
            "timestamp": datetime.now().isoformat()
        }))
        
        # Plan execution with enhanced workflow awareness
        plan = await planner.plan(request.query)
        
        # Execute DAG
        results = await orchestrator.execute_dag(plan.dag, trace_id)
        
        # Compose final result
        final_result = composer.compose_results(results, plan.intent)
        
        # Calculate execution time
        execution_time = round(time.time() - start_time, 2)
        
        # Save execution history
        execution_data = {
            "trace_id": trace_id,
            "query": request.query,
            "intent": plan.intent,
            "workflow_id": getattr(plan, 'workflow_id', None),
            "status": "completed",
            "started_at": datetime.fromtimestamp(start_time).isoformat(),
            "completed_at": datetime.now().isoformat(),
            "duration_seconds": execution_time,
            "node_count": len(plan.dag.nodes),
            "success_count": sum(1 for r in results if r.success),
            "error_count": sum(1 for r in results if not r.success),
            "results": [r.dict() for r in results],
            "final_result": final_result,
            "metadata": {
                "plan_complexity": plan.complexity,
                "estimated_duration": plan.estimated_duration,
                "actual_duration": execution_time
            }
        }
        
        # Save to database in background
        background_tasks.add_task(db_manager.save_execution_history, execution_data)
        
        # Broadcast execution completion
        await connection_manager.broadcast(json.dumps({
            "type": "execution_complete",
            "trace_id": trace_id,
            "success": all(r.success for r in results),
            "duration": execution_time,
            "timestamp": datetime.now().isoformat()
        }))
        
        response = QueryResponse(
            trace_id=trace_id,
            intent=plan.intent,
            results=[r.dict() for r in results],
            final_result=final_result,
            success=all(r.success for r in results),
            execution_time=execution_time,
            plan_metadata={
                "complexity": plan.complexity,
                "node_count": len(plan.dag.nodes),
                "estimated_duration": plan.estimated_duration
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Query execution failed: {e}")
        
        # Broadcast execution error
        await connection_manager.broadcast(json.dumps({
            "type": "execution_error",
            "trace_id": trace_id if 'trace_id' in locals() else "unknown",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }))
        
        raise HTTPException(status_code=500, detail=str(e))

# Enhanced Workflow Management Endpoints

@app.get("/api/workflows")
async def list_workflows(
    author: Optional[str] = Query(None, description="Filter by author"),
    include_public: bool = Query(True, description="Include public workflows"),
    token: str = Depends(verify_token)
):
    """List all available workflows (predefined + user-created)"""
    try:
        # Get workflows from planner's workflow manager
        workflows = await planner.workflow_manager.list_workflows()
        
        # Filter by author if specified
        if author:
            workflows = [wf for wf in workflows if wf.get("author") == author]
        
        # Filter public workflows if needed
        if not include_public:
            workflows = [wf for wf in workflows if wf.get("author") != "system"]
        
        return {
            "success": True,
            "workflows": workflows,
            "count": len(workflows)
        }
        
    except Exception as e:
        logger.error(f"Failed to list workflows: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/workflows")
async def create_workflow(
    workflow_data: Dict[str, Any],
    token: str = Depends(verify_token)
):
    """Create a new user-defined workflow"""
    try:
        result = await planner.workflow_manager.create_workflow(workflow_data)
        
        if result["success"]:
            return {
                "success": True,
                "workflow_id": result["workflow_id"],
                "message": result["message"]
            }
        else:
            raise HTTPException(status_code=400, detail=result["error"])
            
    except Exception as e:
        logger.error(f"Failed to create workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/workflows/{workflow_id}")
async def get_workflow(
    workflow_id: str,
    token: str = Depends(verify_token)
):
    """Get workflow details by ID"""
    try:
        # Try user workflows first
        workflow = await planner.workflow_manager.workflow_manager.get_user_workflow(workflow_id)
        
        if not workflow:
            # Try predefined workflows
            predefined = registry.get_workflow(workflow_id)
            if predefined:
                workflow = {
                    "id": predefined.id,
                    "name": predefined.name,
                    "description": predefined.description,
                    "type": "predefined",
                    "nodes": [node.dict() for node in predefined.plan.nodes],
                    "edges": predefined.plan.edges,
                    "metadata": predefined.metadata
                }
        
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        return {"success": True, "workflow": workflow}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/workflows/{workflow_id}")
async def update_workflow(
    workflow_id: str,
    updates: Dict[str, Any],
    token: str = Depends(verify_token)
):
    """Update an existing workflow"""
    try:
        result = await planner.workflow_manager.update_workflow(workflow_id, updates)
        
        if result["success"]:
            return {"success": True, "message": result["message"]}
        else:
            raise HTTPException(status_code=400, detail=result["error"])
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update workflow {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/workflows/{workflow_id}")
async def delete_workflow(
    workflow_id: str,
    token: str = Depends(verify_token)
):
    """Delete a user-defined workflow"""
    try:
        result = await planner.workflow_manager.delete_workflow(workflow_id)
        
        if result["success"]:
            return {"success": True, "message": result["message"]}
        else:
            raise HTTPException(status_code=404, detail=result["error"])
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete workflow {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/workflows/{workflow_id}/run")
async def run_workflow(
    workflow_id: str,
    parameters: Optional[Dict[str, Any]] = None,
    background_tasks: BackgroundTasks = None,
    token: str = Depends(verify_token)
):
    """Execute a specific workflow by ID"""
    try:
        trace_id = f"workflow_{workflow_id}_{int(time.time() * 1000)}"
        start_time = time.time()
        
        # Get workflow
        workflow = await planner.workflow_manager.workflow_manager.get_user_workflow(workflow_id)
        if not workflow:
            # Try predefined workflows
            predefined = registry.get_workflow(workflow_id)
            if not predefined:
                raise HTTPException(status_code=404, detail="Workflow not found")
            workflow = predefined
        
        logger.info(f"Executing workflow [{trace_id}]: {workflow_id}")
        
        # Broadcast workflow start
        await connection_manager.broadcast(json.dumps({
            "type": "workflow_start",
            "trace_id": trace_id,
            "workflow_id": workflow_id,
            "timestamp": datetime.now().isoformat()
        }))
        
        # Execute workflow
        if isinstance(workflow, dict):
            # User-created workflow
            dag = planner.plan_generator._convert_user_workflow_to_dag(workflow, f"Execute workflow: {workflow_id}")
        else:
            # Predefined workflow
            dag = workflow.plan
        
        results = await orchestrator.execute_dag(dag, trace_id)
        final_result = composer.compose_results(results, workflow.get("intent", "workflow"))
        
        execution_time = round(time.time() - start_time, 2)
        success = all(r.success for r in results)
        
        # Update workflow statistics
        if isinstance(workflow, dict):
            background_tasks.add_task(
                db_manager.update_workflow_stats,
                workflow_id, trace_id, execution_time, success
            )
        
        # Save execution history
        execution_data = {
            "trace_id": trace_id,
            "query": f"Execute workflow: {workflow_id}",
            "intent": "workflow_execution",
            "workflow_id": workflow_id,
            "status": "completed",
            "started_at": datetime.fromtimestamp(start_time).isoformat(),
            "completed_at": datetime.now().isoformat(),
            "duration_seconds": execution_time,
            "node_count": len(dag.nodes),
            "success_count": sum(1 for r in results if r.success),
            "error_count": sum(1 for r in results if not r.success),
            "results": [r.dict() for r in results],
            "final_result": final_result,
            "metadata": {"workflow_execution": True}
        }
        
        background_tasks.add_task(db_manager.save_execution_history, execution_data)
        
        # Broadcast workflow completion
        await connection_manager.broadcast(json.dumps({
            "type": "workflow_complete",
            "trace_id": trace_id,
            "workflow_id": workflow_id,
            "success": success,
            "duration": execution_time,
            "timestamp": datetime.now().isoformat()
        }))
        
        return {
            "success": True,
            "trace_id": trace_id,
            "workflow_id": workflow_id,
            "results": [r.dict() for r in results],
            "final_result": final_result,
            "execution_time": execution_time
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Enhanced Admin Endpoints

@app.post("/api/admin/sql")
async def execute_sql(
    query_data: Dict[str, Any],
    token: str = Depends(verify_token)
):
    """Execute raw SQL query with enhanced security and pagination"""
    try:
        result = await db_manager.execute_raw_sql(
            query=query_data.get("query", "").strip(),
            params=query_data.get("params"),
            page=query_data.get("page", 1),
            page_size=query_data.get("page_size", 50)
        )
        
        return result
        
    except Exception as e:
        logger.error(f"SQL execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/admin/stats")
async def get_system_stats(
    days: int = Query(7, description="Number of days for statistics"),
    token: str = Depends(verify_token)
):
    """Get comprehensive system statistics"""
    try:
        # Get agent performance stats
        agent_stats = await db_manager.get_agent_stats(days)
        
        # Get execution history summary
        execution_history = await db_manager.get_execution_history(limit=10)
        
        # Get workflow statistics
        workflows = await planner.workflow_manager.list_workflows()
        
        # Calculate summary metrics
        total_executions = len(execution_history)
        successful_executions = sum(1 for exec in execution_history if exec.get("success_count", 0) > exec.get("error_count", 0))
        avg_execution_time = sum(exec.get("duration_seconds", 0) for exec in execution_history) / max(total_executions, 1)
        
        return {
            "success": True,
            "statistics": {
                "execution_summary": {
                    "total_executions": total_executions,
                    "successful_executions": successful_executions,
                    "success_rate": round((successful_executions / max(total_executions, 1)) * 100, 1),
                    "avg_execution_time": round(avg_execution_time, 2)
                },
                "agent_performance": agent_stats,
                "workflow_summary": {
                    "total_workflows": len(workflows),
                    "user_workflows": len([wf for wf in workflows if wf.get("type") == "user_created"]),
                    "predefined_workflows": len([wf for wf in workflows if wf.get("type") == "predefined"])
                },
                "recent_executions": execution_history[:5]
            },
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/admin/execution-history")
async def get_execution_history(
    limit: int = Query(50, description="Number of records to return"),
    offset: int = Query(0, description="Offset for pagination"),
    token: str = Depends(verify_token)
):
    """Get execution history with pagination"""
    try:
        history = await db_manager.get_execution_history(limit, offset)
        
        return {
            "success": True,
            "history": history,
            "limit": limit,
            "offset": offset,
            "count": len(history)
        }
        
    except Exception as e:
        logger.error(f"Failed to get execution history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Agent and Tool Management

@app.get("/api/agents")
async def list_agents(token: str = Depends(verify_token)):
    """List all available agents and their tools"""
    try:
        agents = registry.list_agents()
        agent_data = []
        
        for agent in agents:
            tools_data = []
            for tool in agent.tools:
                tools_data.append({
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.params_schema
                })
            
            agent_data.append({
                "name": agent.name,
                "description": agent.description,
                "endpoint": agent.endpoint,
                "enabled": agent.enabled,
                "tools": tools_data,
                "tool_count": len(tools_data)
            })
        
        return {
            "success": True,
            "agents": agent_data,
            "total_agents": len(agent_data),
            "enabled_agents": len([a for a in agent_data if a["enabled"]])
        }
        
    except Exception as e:
        logger.error(f"Failed to list agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/agents/{agent_name}/tools")
async def list_agent_tools(
    agent_name: str,
    token: str = Depends(verify_token)
):
    """List tools for a specific agent"""
    try:
        agent = registry.get_agent(agent_name)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        tools_data = []
        for tool in agent.tools:
            tools_data.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.params_schema,
                "returns": getattr(tool, 'returns_schema', {})
            })
        
        return {
            "success": True,
            "agent_name": agent_name,
            "tools": tools_data,
            "tool_count": len(tools_data)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list tools for agent {agent_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time execution updates"""
    await connection_manager.connect(websocket)
    try:
        # Send connection confirmation
        await websocket.send_text(json.dumps({
            "type": "connected",
            "message": "Connected to Agentic Framework",
            "timestamp": datetime.now().isoformat()
        }))
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle different message types
                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    }))
                elif message.get("type") == "subscribe":
                    # Handle subscription requests
                    await websocket.send_text(json.dumps({
                        "type": "subscribed",
                        "subscription": message.get("subscription", "all"),
                        "timestamp": datetime.now().isoformat()
                    }))
                
            except asyncio.TimeoutError:
                # Send periodic heartbeat
                await websocket.send_text(json.dumps({
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat()
                }))
            
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        connection_manager.disconnect(websocket)

# Health check endpoint
@app.get("/api/health")
async def health_check():
    """System health check"""
    try:
        # Check database connection
        db_health = "healthy"
        try:
            await db_manager.get_system_setting("health_check", "ok")
        except:
            db_health = "error"
        
        # Check MCP client connections
        mcp_health = "healthy" if mcp_client_manager.get_active_connections() else "no_agents"
        
        # Check agent availability
        agents = registry.list_agents()
        enabled_agents = [a for a in agents if a.enabled]
        
        return {
            "status": "healthy",
            "database": db_health,
            "mcp_clients": mcp_health,
            "agents": {
                "total": len(agents),
                "enabled": len(enabled_agents),
                "status": "healthy" if enabled_agents else "no_agents"
            },
            "uptime": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Store start time for uptime calculation
@app.on_event("startup")
async def store_start_time():
    app.state.start_time = time.time()

if __name__ == "__main__":
    import uvicorn
    
    logger.info("=" * 60)
    logger.info("🚀 Starting Enhanced Agentic Framework")
    logger.info("=" * 60)
    logger.info(f"🌐 Host: {settings.host}")
    logger.info(f"🔌 Port: {settings.port}")
    logger.info(f"📊 Features: Enhanced Workflow Management, Real-time Updates, Advanced Admin")
    logger.info(f"🤖 AI Integration: Azure OpenAI powered intelligence")
    logger.info(f"🔐 Security: Bearer token authentication")
    logger.info("=" * 60)
    
    uvicorn.run(
        "agentic.app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.app_env == "dev",
        log_level=settings.log_level.lower()
    )