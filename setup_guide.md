# Complete Agentic Framework Setup Guide

## 📁 Complete Directory Structure

Create the following directory structure on your Windows machine:

```
agentic-framework/
│
├── agentic/
│   ├── __init__.py
│   ├── cli.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── types.py
│   │   ├── config.py
│   │   ├── registry.py
│   │   ├── planner.py
│   │   ├── orchestrator.py
│   │   ├── mcp_client.py
│   │   ├── llm_client.py
│   │   └── composer.py
│   │
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── static/
│   │       └── index.html
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── web_search.yaml
│   │   ├── tabulator.yaml
│   │   ├── nlp_summarizer.yaml
│   │   ├── calculator.yaml
│   │   └── local/
│   │       ├── __init__.py
│   │       ├── web_search_server.py
│   │       ├── tabulator_server.py
│   │       ├── nlp_summarizer_server.py
│   │       └── calculator_server.py
│   │
│   ├── prompts/
│   │   ├── intent/
│   │   │   └── classifier.md
│   │   ├── planner/
│   │   │   └── plan_builder.md
│   │   └── composer/
│   │       └── final_answer.md
│   │
│   └── workflows/
│       ├── market_research_shoes.yaml
│       ├── simple_search.yaml
│       └── data_analysis.yaml
│
├── scripts/
│   ├── setup.bat
│   ├── setup.sh
│   ├── dev_run.bat
│   ├── dev_run.sh
│   ├── test.sh
│   └── build.sh
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_registry.py
│   ├── test_planner.py
│   ├── test_orchestrator.py
│   └── integration/
│       └── test_full_workflow.py
│
├── logs/
│   └── (log files will be created here)
│
├── .env.example
├── .env (create from .env.example)
├── requirements.txt
├── pyproject.toml
├── setup.py
└── README.md
```

## 🚀 Step-by-Step Setup (Windows)

### 1. Initial Setup

```bash
# Create project directory
mkdir agentic-framework
cd agentic-framework

# Create all the files from the artifacts above
# (Copy content from each artifact to corresponding files)

# Make setup script executable and run it
scripts\setup.bat
```

### 2. Configure Environment

Edit `.env` file with your Azure OpenAI credentials:

```env
# Required: Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_DEPLOYMENT=gpt-4
AZURE_OPENAI_API_VERSION=2024-12-01-preview

# Application Settings
HOST=0.0.0.0
PORT=8080
AUTH_TOKEN=devtoken123

# Optional: For enhanced search capabilities
BING_API_KEY=your-bing-key-if-available
```

### 3. Test the Installation

```bash
# Activate virtual environment
venv\Scripts\activate

# Validate configuration
python -m agentic.cli validate

# List available agents
python -m agentic.cli agents

# List available workflows
python -m agentic.cli workflows
```

### 4. Start the Framework

```bash
# Start all services (recommended for first time)
scripts\dev_run.bat
```

This will:
- Start 4 MCP agent servers (ports 9101-9104)
- Start the main FastAPI server (port 8080)
- Open web interface at http://localhost:8080

### 5. Test the Web Interface

1. Open browser to http://localhost:8080
2. You should see the Agentic Framework web interface
3. Try executing a query: "Do market research on running shoes"
4. Watch the workflow execute in real-time

## 🧪 Testing Your Setup

### Quick Test Commands

```bash
# Test CLI query execution
python -m agentic.cli run "Search for Python tutorials"

# Test market research workflow
python -m agentic.cli run "Do market research on electric vehicles" --workflow market_research_shoes

# Test with verbose output
python -m agentic.cli run "Calculate compound interest" --verbose
```

### Web Interface Tests

1. **Workflow Execution Tab**:
   - Enter: "Do market research on smartphones"
   - Click "Execute Workflow"
   - Should see execution status and results

2. **Agents & Tools Tab**:
   - Should show 4 agents: web_search, tabulator, nlp_summarizer, calculator
   - Each agent should show available tools

3. **Admin Tab**:
   - Try SQL query: `SELECT name FROM sqlite_master WHERE type='table';`
   - Should execute without errors

## 🔧 Customization Examples

### Add a New Agent

1. **Create the MCP server** (`agentic/agents/local/email_agent_server.py`):

```python
from fastmcp import FastMCP

mcp = FastMCP("Email Agent")

@mcp.tool()
def send_email(to: str, subject: str, body: str) -> dict:
    """Send an email (mock implementation)"""
    return {
        "success": True,
        "message": f"Email sent to {to}",
        "subject": subject
    }

if __name__ == "__main__":
    mcp.run(port=9105)
```

2. **Create agent config** (`agentic/agents/email_agent.yaml`):

```yaml
name: email_agent
endpoint: http://localhost:9105
description: Email sending and management
tools:
  - name: send_email
    description: Send an email
    params_schema:
      type: object
      properties:
        to: {type: string}
        subject: {type: string}
        body: {type: string}
      required: [to, subject, body]
```

3. **Restart framework** - new agent auto-discovered!

### Create a Custom Workflow

Create `agentic/workflows/product_analysis.yaml`:

```yaml
id: product_analysis
name: Product Analysis Workflow
description: Comprehensive product research and analysis
intent: product_analysis
plan:
  nodes:
    - id: search_product
      agent: web_search
      tool: search
      params:
        query: "{{inputs.product}} reviews 2024"
        limit: 10
      parallel_group: research
    
    - id: search_competitors
      agent: web_search
      tool: search
      params:
        query: "{{inputs.product}} competitors comparison"
        limit: 8
      parallel_group: research
    
    - id: analyze_findings
      agent: nlp_summarizer
      tool: summarize
      params:
        text: "{{results.research.combined}}"
        style: detailed
      depends_on: [research]
    
    - id: create_report
      agent: tabulator
      tool: tabulate
      params:
        data: "{{results.analyze_findings.data}}"
        fields: ["product", "rating", "price", "pros", "cons"]
      depends_on: [analyze_findings]
```

## 🐛 Troubleshooting

### Common Issues

1. **Port conflicts**:
   ```bash
   # Check if ports are in use
   netstat -an | findstr "9101 9102 9103 9104 8080"
   ```

2. **Module import errors**:
   ```bash
   # Ensure you're in the project root and venv is activated
   cd agentic-framework
   venv\Scripts\activate
   python -c "import agentic; print('Import successful')"
   ```

3. **Agent servers not starting**:
   ```bash
   # Test individual agent
   python agentic/agents/local/web_search_server.py
   # Should show: "Starting Web Search Server on port 9101..."
   ```

4. **Azure OpenAI connection issues**:
   ```bash
   # Test LLM connection
   python -c "
   from agentic.core.llm_client import llm_client
   import asyncio
   async def test():
       result = await llm_client.generate([{'role': 'user', 'content': 'Hello'}])
       print(f'Response: {result}')
   asyncio.run(test())
   "
   ```

### Debug Mode

Enable detailed logging:

```env
# In .env file
LOG_LEVEL=DEBUG
DEBUG=true
```

## 🎯 Example Use Cases

### 1. Market Research Automation

```bash
python -m agentic.cli run "Research the electric vehicle market in Europe, analyze key players, and create a competitive analysis table"
```

### 2. Data Analysis Pipeline

```bash
python -m agentic.cli run "Analyze quarterly sales data, identify trends, and summarize key insights"
```

### 3. Content Research & Summarization

```bash
python -m agentic.cli run "Find recent articles about AI developments, summarize key findings, and tabulate the main technologies mentioned"
```

## 🔄 Next Steps

1. **Explore the web interface** - Try different queries and workflows
2. **Add your own agents** - Create custom tools for your specific needs
3. **Build complex workflows** - Chain multiple agents for sophisticated automation
4. **Integrate with your systems** - Use the API endpoints to integrate with existing tools
5. **Monitor and optimize** - Use the admin panel to track performance and optimize workflows

## 🆘 Getting Help

- **Logs**: Check `logs/agentic.log` for detailed execution logs
- **Admin Panel**: Use the SQL executor to inspect the system state
- **CLI Validation**: Run `python -m agentic.cli validate` to check configuration
- **Verbose Mode**: Add `--verbose` to CLI commands for detailed output

## 🎉 Success Indicators

You know everything is working when:

✅ All 4 agent servers start without errors  
✅ Web interface loads at http://localhost:8080  
✅ Agent & Tools tab shows all 4 agents  
✅ Workflow execution completes successfully  
✅ Results are displayed in the web interface  
✅ CLI commands execute without errors  

## 🚀 Production Deployment

For production use:

1. **Security**: Change AUTH_TOKEN to a secure value
2. **Database**: Switch from SQLite to PostgreSQL/MySQL
3. **Monitoring**: Add proper logging and monitoring
4. **Scaling**: Use multiple worker processes
5. **SSL**: Enable HTTPS with proper certificates

```bash
# Production server start
uvicorn agentic.app.main:app --host 0.0.0.0 --port 8080 --workers 4
```
