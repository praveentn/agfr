# README.md
# Agentic Framework

A powerful multi-agent workflow orchestration system built with Python, FastAPI, and the Model Context Protocol (MCP). This framework enables you to create, configure, and execute complex workflows by orchestrating multiple AI agents and tools.

## 🚀 Features

- **Multi-Agent Orchestration**: Coordinate multiple MCP-compliant agents
- **Dynamic Workflow Planning**: AI-powered intent detection and workflow generation
- **Parallel & Sequential Execution**: Support for complex DAG-based workflows
- **Web Interface**: Clean, enterprise-grade web UI for workflow management
- **Extensible Architecture**: Easily add new agents, tools, and workflows
- **Real-time Monitoring**: Track workflow execution with detailed logging
- **Admin Tools**: SQL executor and system management features

## 🏗️ Architecture

```
Client (Web UI) ──► FastAPI Planner Service
                    │
                    ├─► Registry/Discovery (YAML configs)
                    ├─► Intent Classifier (LLM + rules)
                    ├─► Planner (DAG builder)
                    ├─► Orchestrator (async executor)
                    │   └─► MCP Client Manager
                    │       └─► Sub-Agents (FastMCP servers)
                    └─► Result Composer
```

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- Node.js (for MCP Inspector, optional)
- Azure OpenAI API access

### Quick Setup

#### Windows
```bash
git clone <repository-url>
cd agentic-framework
scripts\setup.bat
```

#### Linux/Mac
```bash
git clone <repository-url>
cd agentic-framework
chmod +x scripts/setup.sh
./scripts/setup.sh
```

### Manual Setup

1. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your Azure OpenAI credentials
   ```

4. **Create directory structure**:
   ```bash
   mkdir -p agentic/{agents,prompts,workflows,app/static} logs tests
   ```

## ⚙️ Configuration

Edit `.env` file with your settings:

```env
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_DEPLOYMENT=gpt-4
AZURE_OPENAI_API_VERSION=2024-12-01-preview

# Application Settings
HOST=0.0.0.0
PORT=8080
AUTH_TOKEN=your-secure-token

# Optional: Search APIs
BING_API_KEY=your-bing-key
GOOGLE_API_KEY=your-google-key
```

## 🚀 Usage

### Start the Framework

#### Development Mode
```bash
# Windows
scripts\dev_run.bat

# Linux/Mac
./scripts/dev_run.sh
```

This starts:
- MCP agent servers (ports 9101-9104)
- Main FastAPI server (port 8080)
- Web interface at http://localhost:8080

#### Manual Start
```bash
# Start agents in separate terminals
python agentic/agents/local/web_search_server.py
python agentic/agents/local/tabulator_server.py
python agentic/agents/local/nlp_summarizer_server.py
python agentic/agents/local/calculator_server.py

# Start main server
python -m agentic.app.main
```

### Command Line Interface

```bash
# List available agents and tools
python -m agentic.cli agents

# List available workflows
python -m agentic.cli workflows

# Execute a query
python -m agentic.cli run "Do market research on running shoes"

# Run specific workflow
python -m agentic.cli run "Custom query" --workflow market_research_shoes

# Start server
python -m agentic.cli server --port 8080

# Validate configuration
python -m agentic.cli validate
```

### Web Interface

Open http://localhost:8080 to access:

1. **Workflow Execution**: Execute queries and view results
2. **Agents & Tools**: Browse available agents and their capabilities
3. **Saved Workflows**: Manage predefined workflows
4. **Admin Panel**: SQL executor and system management

## 🔧 Adding New Components

### New MCP Agent

1. **Create server** (e.g., `agentic/agents/local/my_agent_server.py`):
   ```python
   from mcp.server.fastmcp import FastMCP
   
   mcp = FastMCP("My Agent")
   
   @mcp.tool()
   def my_tool(param1: str, param2: int = 10) -> dict:
       """My custom tool"""
       return {"result": f"Processed {param1} with {param2}"}
   
   if __name__ == "__main__":
       mcp.run(port=9105)
   ```

2. **Create config** (`agentic/agents/my_agent.yaml`):
   ```yaml
   name: my_agent
   endpoint: http://localhost:9105
   description: My custom agent
   tools:
     - name: my_tool
       description: My custom tool
       params_schema:
         type: object
         properties:
           param1: {type: string}
           param2: {type: integer, default: 10}
         required: [param1]
   ```

3. **Restart framework** - agent auto-discovered!

### New Workflow

Create `agentic/workflows/my_workflow.yaml`:

```yaml
id: my_workflow
name: My Custom Workflow
description: Custom workflow description
intent: my_intent
plan:
  nodes:
    - id: step1
      agent: web_search
      tool: search
      params:
        query: "{{inputs.topic}} trends 2024"
        limit: 5
      parallel_group: research
    
    - id: step2
      agent: my_agent
      tool: my_tool
      params:
        param1: "{{results.step1.data}}"
      depends_on: [research]
```

### New Prompt Template

Create `agentic/prompts/category/my_prompt.md`:

```markdown
# My Custom Prompt

You are an expert assistant for...

## Instructions:
1. Analyze the input
2. Generate insights
3. Format output as JSON

## Example:
Input: "market research request"
Output: {"insights": ["insight1", "insight2"]}
```

## 📊 Example Workflows

### Market Research
```
Query: "Do market research on electric vehicles"

Execution Plan:
1. Parallel web searches for EV market trends, competitors, pricing
2. Summarize findings using NLP agent
3. Create structured table with tabulator agent
4. Compose final research report
```

### Data Analysis
```
Query: "Analyze this sales data and create a summary"

Execution Plan:
1. Process data with analyzer agent
2. Extract key metrics and insights
3. Generate visualizations (if configured)
4. Create executive summary
```

## 🧪 Testing

```bash
# Run all tests
./scripts/test.sh

# Run specific test categories
python -m pytest tests/unit/ -v
python -m pytest tests/integration/ -v

# Run with coverage
python -m pytest --cov=agentic tests/
```

## 📁 Project Structure

```
agentic/
├── core/                   # Core framework components
│   ├── types.py           # Pydantic models
│   ├── config.py          # Configuration management
│   ├── registry.py        # Agent/workflow discovery
│   ├── planner.py         # Intent classification & planning
│   ├── orchestrator.py    # DAG execution engine
│   ├── mcp_client.py      # MCP protocol client
│   └── composer.py        # Result composition
├── app/                   # FastAPI application
│   ├── main.py           # API endpoints
│   └── static/           # Web interface
├── agents/               # Agent configurations
│   ├── *.yaml           # Agent specifications
│   └── local/           # Local MCP servers
├── workflows/           # Workflow definitions
├── prompts/            # Prompt templates
└── scripts/            # Setup and utility scripts
```

## 🔒 Security

- Bearer token authentication for API access
- SQL injection protection for admin queries
- Configurable agent permissions
- Request/response logging and monitoring

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

1. **Agents not starting**: Check port availability (9101-9104)
2. **Azure OpenAI errors**: Verify API key and endpoint in `.env`
3. **Permission errors**: Ensure proper file permissions on scripts
4. **Import errors**: Activate virtual environment and install dependencies

### Getting Help

- Check the logs in `logs/agentic.log`
- Use the admin panel's SQL executor to inspect data
- Run `python -m agentic.cli validate` to check configuration
- Enable verbose logging with `LOG_LEVEL=DEBUG`

## 🔮 Roadmap

- [ ] Visual workflow builder (drag-and-drop)
- [ ] More built-in agent types (email, calendar, databases)
- [ ] Workflow templates marketplace
- [ ] Advanced monitoring and analytics
- [ ] Multi-tenant support
- [ ] Kubernetes deployment manifests
- [ ] Integration with popular AI platforms

---

**Built with ❤️ using FastAPI, FastMCP, and the Model Context Protocol**