# agentic/agents/local/tabulator_server.py
import asyncio
import json
import re
from typing import Dict, List, Any, Union

import pandas as pd
from fastmcp import FastMCP

mcp = FastMCP("Tabulator Server")

@mcp.tool()
def tabulate(data: Union[str, List, Dict], fields: List[str] = None, format: str = "json") -> Dict[str, Any]:
    """
    Convert input data into structured table format (JSON, CSV, or HTML).
    Accepts raw text, JSON, dicts, or lists of dicts.
    """
    try:
        # Parse input data
        if isinstance(data, str):
            try:
                parsed = json.loads(data)
            except Exception:
                parsed = _extract_data_from_text(data, fields)
        else:
            parsed = data

        # Normalize to list of dicts
        if isinstance(parsed, dict):
            table_data = parsed.get("items", [parsed])
        elif isinstance(parsed, list):
            table_data = parsed
        else:
            table_data = [{"value": str(parsed)}]

        # Filter fields if specified
        if fields:
            table_data = [
                {f: (row.get(f, "N/A") if isinstance(row, dict) else "N/A") for f in fields}
                for row in table_data
            ]

        # Output formats
        if format.lower() == "csv" and table_data and isinstance(table_data[0], dict):
            df = pd.DataFrame(table_data)
            csv_output = df.to_csv(index=False)
            return {
                "table": csv_output,
                "format": "csv",
                "row_count": len(table_data),
                "columns": list(df.columns)
            }

        if format.lower() == "html" and table_data and isinstance(table_data[0], dict):
            df = pd.DataFrame(table_data)
            html_output = df.to_html(index=False, classes="table table-striped")
            return {
                "table": html_output,
                "format": "html",
                "row_count": len(table_data),
                "columns": list(df.columns)
            }

        # Default JSON
        return {
            "table": table_data,
            "format": "json",
            "row_count": len(table_data),
            "columns": list(table_data[0].keys()) if table_data and isinstance(table_data[0], dict) else []
        }

    except Exception as e:
        return {
            "table": [],
            "error": str(e),
            "format": format,
            "row_count": 0
        }

def _extract_data_from_text(text: str, fields: List[str] = None) -> List[Dict]:
    """Extract structured rows from plain text lines"""
    rows = []
    for line in text.strip().splitlines():
        if not line.strip():
            continue
        if fields:
            parts = re.split(r'[,;:|]', line)
            row = {f: (parts[i].strip() if i < len(parts) else "N/A") for i, f in enumerate(fields)}
            rows.append(row)
        else:
            if ":" in line:
                k, v = line.split(":", 1)
                rows.append({"key": k.strip(), "value": v.strip()})
            else:
                rows.append({"content": line.strip()})
    return rows

async def main():
    print("Starting Tabulator Server on port 9102 (async mode)...")
    await mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=9102,
        log_level="info"
    )

if __name__ == "__main__":
    asyncio.run(main())
