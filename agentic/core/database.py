# agentic/core/database.py
import sqlite3
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import aiosqlite

from .config import settings

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Enhanced database manager with workflow support"""
    
    def __init__(self):
        self.db_path = "agentic.db"
        self.initialized = False
    
    async def initialize(self):
        """Initialize database with required tables"""
        if self.initialized:
            return
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Enable foreign keys
                await db.execute("PRAGMA foreign_keys = ON")
                
                # Create execution history table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS execution_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        trace_id TEXT UNIQUE NOT NULL,
                        query TEXT NOT NULL,
                        intent TEXT NOT NULL,
                        workflow_id TEXT,
                        status TEXT NOT NULL DEFAULT 'running',
                        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        completed_at TIMESTAMP,
                        duration_seconds REAL,
                        node_count INTEGER DEFAULT 0,
                        success_count INTEGER DEFAULT 0,
                        error_count INTEGER DEFAULT 0,
                        results_json TEXT,
                        final_result_json TEXT,
                        metadata_json TEXT
                    )
                """)
                
                # Create user workflows table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS user_workflows (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        description TEXT,
                        author TEXT DEFAULT 'user',
                        tags_json TEXT DEFAULT '[]',
                        nodes_json TEXT NOT NULL,
                        edges_json TEXT DEFAULT '[]',
                        metadata_json TEXT DEFAULT '{}',
                        is_public BOOLEAN DEFAULT FALSE,
                        is_active BOOLEAN DEFAULT TRUE,
                        execution_count INTEGER DEFAULT 0,
                        success_count INTEGER DEFAULT 0,
                        avg_duration_seconds REAL DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_executed_at TIMESTAMP
                    )
                """)
                
                # Create workflow execution stats table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS workflow_stats (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        workflow_id TEXT NOT NULL,
                        trace_id TEXT NOT NULL,
                        execution_time_seconds REAL,
                        success BOOLEAN,
                        error_message TEXT,
                        executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (workflow_id) REFERENCES user_workflows (id),
                        FOREIGN KEY (trace_id) REFERENCES execution_history (trace_id)
                    )
                """)
                
                # Create agent performance table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS agent_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        agent_name TEXT NOT NULL,
                        tool_name TEXT NOT NULL,
                        trace_id TEXT NOT NULL,
                        node_id TEXT NOT NULL,
                        status TEXT NOT NULL,
                        started_at TIMESTAMP,
                        completed_at TIMESTAMP,
                        duration_seconds REAL,
                        retry_count INTEGER DEFAULT 0,
                        error_message TEXT,
                        params_json TEXT,
                        result_json TEXT,
                        FOREIGN KEY (trace_id) REFERENCES execution_history (trace_id)
                    )
                """)
                
                # Create system settings table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS system_settings (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        description TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes for performance
                await db.execute("CREATE INDEX IF NOT EXISTS idx_execution_history_trace_id ON execution_history(trace_id)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_execution_history_started_at ON execution_history(started_at)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_user_workflows_name ON user_workflows(name)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_user_workflows_author ON user_workflows(author)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_workflow_stats_workflow_id ON workflow_stats(workflow_id)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_agent_performance_agent_name ON agent_performance(agent_name)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_agent_performance_trace_id ON agent_performance(trace_id)")
                
                await db.commit()
                
            self.initialized = True
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    async def save_execution_history(self, execution_data: Dict[str, Any]) -> bool:
        """Save execution history to database"""
        try:
            await self.initialize()
            
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO execution_history (
                        trace_id, query, intent, workflow_id, status, started_at,
                        completed_at, duration_seconds, node_count, success_count,
                        error_count, results_json, final_result_json, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    execution_data.get("trace_id"),
                    execution_data.get("query"),
                    execution_data.get("intent"),
                    execution_data.get("workflow_id"),
                    execution_data.get("status", "completed"),
                    execution_data.get("started_at"),
                    execution_data.get("completed_at"),
                    round(execution_data.get("duration_seconds", 0), 2),
                    execution_data.get("node_count", 0),
                    execution_data.get("success_count", 0),
                    execution_data.get("error_count", 0),
                    json.dumps(execution_data.get("results", [])),
                    json.dumps(execution_data.get("final_result", {})),
                    json.dumps(execution_data.get("metadata", {}))
                ))
                await db.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to save execution history: {e}")
            return False
    
    async def save_user_workflow(self, workflow_data: Dict[str, Any]) -> bool:
        """Save or update user-defined workflow"""
        try:
            await self.initialize()
            
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO user_workflows (
                        id, name, description, author, tags_json, nodes_json,
                        edges_json, metadata_json, is_public, is_active,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    workflow_data["id"],
                    workflow_data["name"],
                    workflow_data.get("description", ""),
                    workflow_data.get("author", "user"),
                    json.dumps(workflow_data.get("tags", [])),
                    json.dumps(workflow_data["nodes"]),
                    json.dumps(workflow_data.get("edges", [])),
                    json.dumps(workflow_data.get("metadata", {})),
                    workflow_data.get("is_public", False),
                    workflow_data.get("is_active", True),
                    datetime.now().isoformat()
                ))
                await db.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to save workflow: {e}")
            return False
    
    async def get_user_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get user workflow by ID"""
        try:
            await self.initialize()
            
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute("""
                    SELECT * FROM user_workflows WHERE id = ? AND is_active = TRUE
                """, (workflow_id,)) as cursor:
                    row = await cursor.fetchone()
                    
                    if row:
                        return {
                            "id": row[0],
                            "name": row[1],
                            "description": row[2],
                            "author": row[3],
                            "tags": json.loads(row[4] or "[]"),
                            "nodes": json.loads(row[5]),
                            "edges": json.loads(row[6] or "[]"),
                            "metadata": json.loads(row[7] or "{}"),
                            "is_public": row[8],
                            "is_active": row[9],
                            "execution_count": row[10],
                            "success_count": row[11],
                            "avg_duration_seconds": round(row[12] or 0, 2),
                            "created_at": row[13],
                            "updated_at": row[14],
                            "last_executed_at": row[15]
                        }
            return None
            
        except Exception as e:
            logger.error(f"Failed to get workflow {workflow_id}: {e}")
            return None
    
    async def get_user_workflows(self, author: Optional[str] = None, include_public: bool = True) -> List[Dict[str, Any]]:
        """Get all user workflows"""
        try:
            await self.initialize()
            
            query = "SELECT * FROM user_workflows WHERE is_active = TRUE"
            params = []
            
            if author:
                if include_public:
                    query += " AND (author = ? OR is_public = TRUE)"
                    params.append(author)
                else:
                    query += " AND author = ?"
                    params.append(author)
            elif include_public:
                query += " AND is_public = TRUE"
            
            query += " ORDER BY updated_at DESC"
            
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(query, params) as cursor:
                    rows = await cursor.fetchall()
                    
                    workflows = []
                    for row in rows:
                        workflows.append({
                            "id": row[0],
                            "name": row[1],
                            "description": row[2],
                            "author": row[3],
                            "tags": json.loads(row[4] or "[]"),
                            "nodes": json.loads(row[5]),
                            "edges": json.loads(row[6] or "[]"),
                            "metadata": json.loads(row[7] or "{}"),
                            "is_public": row[8],
                            "is_active": row[9],
                            "execution_count": row[10],
                            "success_count": row[11],
                            "avg_duration_seconds": round(row[12] or 0, 2),
                            "created_at": row[13],
                            "updated_at": row[14],
                            "last_executed_at": row[15]
                        })
                    
                    return workflows
                    
        except Exception as e:
            logger.error(f"Failed to get user workflows: {e}")
            return []
    
    async def delete_user_workflow(self, workflow_id: str) -> bool:
        """Delete user workflow (soft delete)"""
        try:
            await self.initialize()
            
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute("""
                    UPDATE user_workflows SET is_active = FALSE, updated_at = ?
                    WHERE id = ?
                """, (datetime.now().isoformat(), workflow_id))
                
                await db.commit()
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"Failed to delete workflow {workflow_id}: {e}")
            return False
    
    async def update_workflow_stats(self, workflow_id: str, trace_id: str, 
                                   execution_time: float, success: bool, 
                                   error_message: Optional[str] = None) -> bool:
        """Update workflow execution statistics"""
        try:
            await self.initialize()
            
            async with aiosqlite.connect(self.db_path) as db:
                # Insert execution record
                await db.execute("""
                    INSERT INTO workflow_stats (
                        workflow_id, trace_id, execution_time_seconds, 
                        success, error_message
                    ) VALUES (?, ?, ?, ?, ?)
                """, (workflow_id, trace_id, round(execution_time, 2), success, error_message))
                
                # Update workflow summary stats
                await db.execute("""
                    UPDATE user_workflows SET 
                        execution_count = execution_count + 1,
                        success_count = success_count + ?,
                        avg_duration_seconds = (
                            SELECT AVG(execution_time_seconds) 
                            FROM workflow_stats 
                            WHERE workflow_id = ?
                        ),
                        last_executed_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (1 if success else 0, workflow_id, workflow_id))
                
                await db.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to update workflow stats: {e}")
            return False
    
    async def save_agent_performance(self, performance_data: Dict[str, Any]) -> bool:
        """Save agent performance metrics"""
        try:
            await self.initialize()
            
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO agent_performance (
                        agent_name, tool_name, trace_id, node_id, status,
                        started_at, completed_at, duration_seconds, retry_count,
                        error_message, params_json, result_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    performance_data.get("agent_name"),
                    performance_data.get("tool_name"),
                    performance_data.get("trace_id"),
                    performance_data.get("node_id"),
                    performance_data.get("status"),
                    performance_data.get("started_at"),
                    performance_data.get("completed_at"),
                    round(performance_data.get("duration_seconds", 0), 3),
                    performance_data.get("retry_count", 0),
                    performance_data.get("error_message"),
                    json.dumps(performance_data.get("params", {})),
                    json.dumps(performance_data.get("result", {}))
                ))
                await db.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to save agent performance: {e}")
            return False
    
    async def get_execution_history(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """Get execution history with pagination"""
        try:
            await self.initialize()
            
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute("""
                    SELECT * FROM execution_history 
                    ORDER BY started_at DESC 
                    LIMIT ? OFFSET ?
                """, (limit, offset)) as cursor:
                    rows = await cursor.fetchall()
                    
                    history = []
                    for row in rows:
                        history.append({
                            "id": row[0],
                            "trace_id": row[1],
                            "query": row[2],
                            "intent": row[3],
                            "workflow_id": row[4],
                            "status": row[5],
                            "started_at": row[6],
                            "completed_at": row[7],
                            "duration_seconds": round(row[8] or 0, 2),
                            "node_count": row[9],
                            "success_count": row[10],
                            "error_count": row[11],
                            "results": json.loads(row[12] or "[]"),
                            "final_result": json.loads(row[13] or "{}"),
                            "metadata": json.loads(row[14] or "{}")
                        })
                    
                    return history
                    
        except Exception as e:
            logger.error(f"Failed to get execution history: {e}")
            return []
    
    async def get_agent_stats(self, days: int = 7) -> Dict[str, Any]:
        """Get agent performance statistics"""
        try:
            await self.initialize()
            
            async with aiosqlite.connect(self.db_path) as db:
                # Get overall stats
                async with db.execute("""
                    SELECT 
                        agent_name,
                        tool_name,
                        COUNT(*) as total_calls,
                        AVG(duration_seconds) as avg_duration,
                        SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as success_count,
                        SUM(retry_count) as total_retries
                    FROM agent_performance 
                    WHERE started_at > datetime('now', '-{} days')
                    GROUP BY agent_name, tool_name
                    ORDER BY total_calls DESC
                """.format(days)) as cursor:
                    rows = await cursor.fetchall()
                    
                    stats = {
                        "agents": [],
                        "summary": {
                            "total_calls": 0,
                            "avg_duration": 0,
                            "success_rate": 0,
                            "total_retries": 0
                        }
                    }
                    
                    total_calls = 0
                    total_duration = 0
                    total_success = 0
                    total_retries = 0
                    
                    for row in rows:
                        agent_stat = {
                            "agent_name": row[0],
                            "tool_name": row[1],
                            "total_calls": row[2],
                            "avg_duration": round(row[3] or 0, 3),
                            "success_count": row[4],
                            "success_rate": round((row[4] / row[2]) * 100, 1) if row[2] > 0 else 0,
                            "total_retries": row[5]
                        }
                        stats["agents"].append(agent_stat)
                        
                        total_calls += row[2]
                        total_duration += (row[3] or 0) * row[2]
                        total_success += row[4]
                        total_retries += row[5]
                    
                    if total_calls > 0:
                        stats["summary"] = {
                            "total_calls": total_calls,
                            "avg_duration": round(total_duration / total_calls, 3),
                            "success_rate": round((total_success / total_calls) * 100, 1),
                            "total_retries": total_retries
                        }
                    
                    return stats
                    
        except Exception as e:
            logger.error(f"Failed to get agent stats: {e}")
            return {"agents": [], "summary": {}}
    
    async def execute_raw_sql(self, query: str, params: Optional[Tuple] = None, 
                             page: int = 1, page_size: int = 50) -> Dict[str, Any]:
        """Execute raw SQL query with pagination"""
        try:
            await self.initialize()
            
            # Security check for dangerous operations
            query_upper = query.upper().strip()
            dangerous_keywords = ["DROP", "DELETE", "TRUNCATE", "ALTER", "CREATE INDEX", "DROP INDEX"]
            is_dangerous = any(keyword in query_upper for keyword in dangerous_keywords)
            
            async with aiosqlite.connect(self.db_path) as db:
                if query_upper.startswith("SELECT"):
                    # Add pagination for SELECT queries
                    offset = (page - 1) * page_size
                    
                    # Count total records first
                    count_query = f"SELECT COUNT(*) FROM ({query})"
                    async with db.execute(count_query, params or ()) as cursor:
                        total_rows = (await cursor.fetchone())[0]
                    
                    # Execute paginated query
                    paginated_query = f"{query} LIMIT {page_size} OFFSET {offset}"
                    async with db.execute(paginated_query, params or ()) as cursor:
                        columns = [description[0] for description in cursor.description]
                        rows = await cursor.fetchall()
                        
                        # Convert rows to dictionaries
                        data = []
                        for row in rows:
                            row_dict = {}
                            for i, value in enumerate(row):
                                if isinstance(value, float):
                                    row_dict[columns[i]] = round(value, 3)
                                else:
                                    row_dict[columns[i]] = value
                            data.append(row_dict)
                        
                        return {
                            "success": True,
                            "data": data,
                            "columns": columns,
                            "total_rows": total_rows,
                            "page": page,
                            "page_size": page_size,
                            "total_pages": (total_rows + page_size - 1) // page_size,
                            "is_dangerous": is_dangerous,
                            "query_type": "SELECT"
                        }
                else:
                    # Execute non-SELECT queries
                    cursor = await db.execute(query, params or ())
                    await db.commit()
                    
                    return {
                        "success": True,
                        "affected_rows": cursor.rowcount,
                        "lastrowid": cursor.lastrowid,
                        "is_dangerous": is_dangerous,
                        "query_type": "MODIFICATION"
                    }
                    
        except Exception as e:
            logger.error(f"SQL execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "is_dangerous": is_dangerous if 'is_dangerous' in locals() else False
            }
    
    async def get_system_setting(self, key: str, default: Any = None) -> Any:
        """Get system setting value"""
        try:
            await self.initialize()
            
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute("""
                    SELECT value FROM system_settings WHERE key = ?
                """, (key,)) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        try:
                            return json.loads(row[0])
                        except json.JSONDecodeError:
                            return row[0]
            return default
            
        except Exception as e:
            logger.error(f"Failed to get system setting {key}: {e}")
            return default
    
    async def set_system_setting(self, key: str, value: Any, description: str = "") -> bool:
        """Set system setting value"""
        try:
            await self.initialize()
            
            async with aiosqlite.connect(self.db_path) as db:
                value_str = json.dumps(value) if not isinstance(value, str) else value
                await db.execute("""
                    INSERT OR REPLACE INTO system_settings (key, value, description, updated_at)
                    VALUES (?, ?, ?, ?)
                """, (key, value_str, description, datetime.now().isoformat()))
                await db.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to set system setting {key}: {e}")
            return False
    
    async def cleanup_old_data(self, days: int = 30) -> Dict[str, int]:
        """Clean up old execution data"""
        try:
            await self.initialize()
            
            cleanup_stats = {"execution_history": 0, "agent_performance": 0, "workflow_stats": 0}
            
            async with aiosqlite.connect(self.db_path) as db:
                # Clean execution history
                cursor = await db.execute("""
                    DELETE FROM execution_history 
                    WHERE started_at < datetime('now', '-{} days')
                """.format(days))
                cleanup_stats["execution_history"] = cursor.rowcount
                
                # Clean agent performance
                cursor = await db.execute("""
                    DELETE FROM agent_performance 
                    WHERE started_at < datetime('now', '-{} days')
                """.format(days))
                cleanup_stats["agent_performance"] = cursor.rowcount
                
                # Clean workflow stats
                cursor = await db.execute("""
                    DELETE FROM workflow_stats 
                    WHERE executed_at < datetime('now', '-{} days')
                """.format(days))
                cleanup_stats["workflow_stats"] = cursor.rowcount
                
                await db.commit()
                
            logger.info(f"Cleaned up old data: {cleanup_stats}")
            return cleanup_stats
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return {"execution_history": 0, "agent_performance": 0, "workflow_stats": 0}

# Create global database manager instance
db_manager = DatabaseManager()