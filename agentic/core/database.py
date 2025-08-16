# ============================================================================
# agentic/core/database.py
import sqlite3
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from .config import settings

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.db_path = settings.database_url.replace("sqlite:///", "")
        self._ensure_database_exists()
        self._create_tables()
    
    def _ensure_database_exists(self):
        """Ensure database file and directory exist"""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        
        if not os.path.exists(self.db_path):
            # Create empty database file
            with sqlite3.connect(self.db_path) as conn:
                pass
            logger.info(f"Created database at {self.db_path}")
    
    def _create_tables(self):
        """Create necessary tables if they don't exist"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Execution logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS execution_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT NOT NULL,
                    node_id TEXT,
                    agent TEXT,
                    tool TEXT,
                    success BOOLEAN,
                    started_at REAL,
                    finished_at REAL,
                    data TEXT,
                    error TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Workflow executions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS workflow_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT UNIQUE NOT NULL,
                    intent TEXT,
                    query TEXT,
                    workflow_id TEXT,
                    status TEXT,
                    final_result TEXT,
                    execution_time REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP
                )
            """)
            
            # Agents registry table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agents_registry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    endpoint TEXT,
                    description TEXT,
                    enabled BOOLEAN DEFAULT 1,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            logger.info("Database tables created/verified")
    
    def execute_query(self, query: str, params: tuple = ()) -> Dict[str, Any]:
        """Execute SQL query and return results"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute(query, params)
                
                if query.strip().upper().startswith('SELECT'):
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
            logger.error(f"Database query failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

# Global database manager
db_manager = DatabaseManager()
