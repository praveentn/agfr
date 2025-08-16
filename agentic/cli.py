# agentic/cli.py
"""
Command-line interface for the Agentic Framework
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

from .core.config import settings
from .core.registry import registry
from .core.planner import planner
from .core.orchestrator import orchestrator
from .app.main import app


def setup_logging():
    """Setup logging configuration"""
    import logging
    
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format=settings.log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(settings.LOG_FILE) if hasattr(settings, 'LOG_FILE') else logging.NullHandler()
        ]
    )


async def run_query(query: str, workflow_id: str = None) -> Dict[str, Any]:
    """Run a query using the agentic framework"""
    from .core.types import QueryRequest
    
    request = QueryRequest(
        text=query,
        options={"use_workflow": workflow_id} if workflow_id else {},
        context={}
    )
    
    # Create execution plan
    dag = await planner.create_plan(request)
    trace_id = f"cli_{int(asyncio.get_event_loop().time() * 1000)}"
    
    # Execute workflow
    results = await orchestrator.execute_dag(dag, trace_id)
    
    # Compose final result
    from .core.composer import composer
    final_result = composer.compose_results(results, dag.metadata.get("intent", "general"))
    
    return {
        "trace_id": trace_id,
        "intent": dag.metadata.get("intent", "unknown"),
        "plan": dag.dict(),
        "results": [r.dict() for r in results],
        "final_result": final_result,
        "success": all(r.success for r in results)
    }


def cmd_list_agents(args):
    """List all available agents"""
    agents = registry.list_agents()
    
    print(f"\n{'=' * 60}")
    print(f"{'AVAILABLE AGENTS':^60}")
    print(f"{'=' * 60}")
    
    for agent in agents:
        status = "✓ Enabled" if agent.enabled else "✗ Disabled"
        print(f"\n🤖 {agent.name}")
        print(f"   Description: {agent.description}")
        print(f"   Endpoint: {agent.endpoint}")
        print(f"   Status: {status}")
        print(f"   Tools: {len(agent.tools)}")
        
        for tool in agent.tools:
            print(f"     • {tool.name}: {tool.description}")


def cmd_list_workflows(args):
    """List all available workflows"""
    workflows = registry.list_workflows()
    
    print(f"\n{'=' * 60}")
    print(f"{'AVAILABLE WORKFLOWS':^60}")
    print(f"{'=' * 60}")
    
    for workflow in workflows:
        print(f"\n📋 {workflow.name} (ID: {workflow.id})")
        print(f"   Description: {workflow.description}")
        print(f"   Intent: {workflow.intent}")
        print(f"   Nodes: {len(workflow.plan.nodes)}")
        
        for node in workflow.plan.nodes:
            print(f"     • {node.id}: {node.agent}.{node.tool}")


async def cmd_run_query(args):
    """Run a query"""
    print(f"\n🚀 Executing query: {args.query}")
    
    if args.workflow:
        print(f"   Using workflow: {args.workflow}")
    
    try:
        result = await run_query(args.query, args.workflow)
        
        print(f"\n✅ Execution completed successfully!")
        print(f"   Trace ID: {result['trace_id']}")
        print(f"   Intent: {result['intent']}")
        print(f"   Steps: {len(result['results'])}")
        print(f"   Success: {result['success']}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"   Results saved to: {args.output}")
        
        if args.verbose:
            print(f"\n📊 Final Result:")
            print(json.dumps(result['final_result'], indent=2, default=str))
            
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
        sys.exit(1)


def cmd_start_server(args):
    """Start the FastAPI server"""
    import uvicorn
    
    print(f"🌐 Starting Agentic Framework server...")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   Debug: {args.debug}")
    
    uvicorn.run(
        "agentic.app.main:app",
        host=args.host,
        port=args.port,
        reload=args.debug,
        log_level="info" if args.verbose else "warning"
    )


def cmd_start_agents(args):
    """Start MCP agent servers"""
    import subprocess
    import time
    
    agents_to_start = args.agents or ["web_search", "tabulator", "nlp_summarizer", "calculator"]
    processes = []
    
    print(f"🤖 Starting MCP agent servers...")
    
    agent_ports = {
        "web_search": 9101,
        "tabulator": 9102, 
        "nlp_summarizer": 9103,
        "calculator": 9104,
    }
    
    for agent_name in agents_to_start:
        if agent_name in agent_ports:
            port = agent_ports[agent_name]
            script_path = f"agentic/agents/local/{agent_name}_server.py"
            
            print(f"   Starting {agent_name} on port {port}...")
            
            try:
                process = subprocess.Popen([
                    sys.executable, script_path
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                processes.append((agent_name, process))
                time.sleep(1)  # Give server time to start
                
            except Exception as e:
                print(f"   ❌ Failed to start {agent_name}: {e}")
    
    if processes:
        print(f"\n✅ Started {len(processes)} agent servers")
        print("   Press Ctrl+C to stop all servers")
        
        try:
            # Wait for all processes
            while True:
                time.sleep(1)
                # Check if any process died
                for name, process in processes:
                    if process.poll() is not None:
                        print(f"   ⚠️  Agent {name} stopped unexpectedly")
                        
        except KeyboardInterrupt:
            print(f"\n🛑 Stopping all agent servers...")
            for name, process in processes:
                process.terminate()
                print(f"   Stopped {name}")


def cmd_validate_config(args):
    """Validate configuration"""
    print(f"\n🔍 Validating configuration...")
    
    # Check required settings
    issues = []
    
    if not settings.azure_openai_api_key:
        issues.append("❌ AZURE_OPENAI_API_KEY not set")
    else:
        print("✅ Azure OpenAI API key configured")
    
    if not settings.azure_openai_endpoint:
        issues.append("❌ AZURE_OPENAI_ENDPOINT not set")
    else:
        print("✅ Azure OpenAI endpoint configured")
    
    # Check directories
    for dir_name in ["agents_dir", "prompts_dir", "workflows_dir"]:
        dir_path = Path(getattr(settings, dir_name))
        if dir_path.exists():
            print(f"✅ {dir_name}: {dir_path}")
        else:
            print(f"⚠️  {dir_name}: {dir_path} (will be created)")
    
    # Check registry
    agents = registry.list_agents()
    workflows = registry.list_workflows()
    
    print(f"✅ Agents loaded: {len(agents)}")
    print(f"✅ Workflows loaded: {len(workflows)}")
    
    if issues:
        print(f"\n❌ Configuration issues found:")
        for issue in issues:
            print(f"   {issue}")
        sys.exit(1)
    else:
        print(f"\n✅ Configuration is valid!")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Agentic Framework - Multi-agent workflow orchestration",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List agents command
    list_agents_parser = subparsers.add_parser("agents", help="List available agents")
    list_agents_parser.set_defaults(func=cmd_list_agents)
    
    # List workflows command  
    list_workflows_parser = subparsers.add_parser("workflows", help="List available workflows")
    list_workflows_parser.set_defaults(func=cmd_list_workflows)
    
    # Run query command
    run_parser = subparsers.add_parser("run", help="Run a query")
    run_parser.add_argument("query", help="Query to execute")
    run_parser.add_argument("--workflow", "-w", help="Specific workflow to use")
    run_parser.add_argument("--output", "-o", help="Output file for results")
    run_parser.set_defaults(func=cmd_run_query)
    
    # Start server command
    server_parser = subparsers.add_parser("server", help="Start FastAPI server")
    server_parser.add_argument("--host", default=settings.host, help="Server host")
    server_parser.add_argument("--port", type=int, default=settings.port, help="Server port")
    server_parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    server_parser.set_defaults(func=cmd_start_server)
    
    # Start agents command
    start_agents_parser = subparsers.add_parser("start-agents", help="Start MCP agent servers")
    start_agents_parser.add_argument("--agents", nargs="+", help="Specific agents to start")
    start_agents_parser.set_defaults(func=cmd_start_agents)
    
    # Validate config command
    validate_parser = subparsers.add_parser("validate", help="Validate configuration")
    validate_parser.set_defaults(func=cmd_validate_config)
    
    args = parser.parse_args()
    
    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit(1)
    
    # Setup logging
    setup_logging()
    
    # Run command
    if asyncio.iscoroutinefunction(args.func):
        asyncio.run(args.func(args))
    else:
        args.func(args)


if __name__ == "__main__":
    main()

