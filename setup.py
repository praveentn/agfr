# setup.py
from setuptools import setup, find_packages

setup(
    name="agentic-framework",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.5.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0.1",
        "aiohttp>=3.9.0",
        "jinja2>=3.1.2",
        "python-multipart>=0.0.6",
        "fastmcp>=0.2.0",
        "openai>=1.3.0",
        "pandas>=2.1.0",
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.0",
        "selenium>=4.15.0",
        "webdriver-manager>=4.0.0",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "agentic=agentic.cli:main",
            "agentic-server=agentic.app.main:main",
        ],
        "agentic.agents": [
            "web_search=agentic.agents.local.web_search_server:mcp",
            "tabulator=agentic.agents.local.tabulator_server:mcp", 
            "nlp_summarizer=agentic.agents.local.nlp_summarizer_server:mcp",
            "calculator=agentic.agents.local.calculator_server:mcp",
        ],
    },
)


