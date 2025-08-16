# agentic/agents/local/tabulator_server.py
from mcp.server.fastmcp import FastMCP
import json
import pandas as pd
from typing import Dict, List, Any, Union
import re

mcp = FastMCP("Tabulator Server")

@mcp.tool()
def tabulate(data: Union[str, List, Dict], fields: List[str] = None, format: str = "json") -> Dict[str, Any]:
    """Convert data into structured table format"""
    try:
        # Parse input data
        if isinstance(data, str):
            # Try to parse as JSON first
            try:
                parsed_data = json.loads(data)
            except:
                # If not JSON, try to extract structured data from text
                parsed_data = _extract_data_from_text(data, fields)
        else:
            parsed_data = data
        
        # Convert to list of dictionaries if needed
        if isinstance(parsed_data, dict):
            if "items" in parsed_data:
                table_data = parsed_data["items"]
            else:
                table_data = [parsed_data]
        elif isinstance(parsed_data, list):
            table_data = parsed_data
        else:
            table_data = [{"value": str(parsed_data)}]
        
        # Filter fields if specified
        if fields:
            filtered_data = []
            for row in table_data:
                if isinstance(row, dict):
                    filtered_row = {field: row.get(field, "N/A") for field in fields}
                    filtered_data.append(filtered_row)
            table_data = filtered_data
        
        # Format output
        if format.lower() == "csv":
            if table_data and isinstance(table_data[0], dict):
                df = pd.DataFrame(table_data)
                csv_output = df.to_csv(index=False)
                return {
                    "table": csv_output,
                    "format": "csv",
                    "row_count": len(table_data),
                    "columns": list(table_data[0].keys()) if table_data else []
                }
        
        elif format.lower() == "html":
            if table_data and isinstance(table_data[0], dict):
                df = pd.DataFrame(table_data)
                html_output = df.to_html(index=False, classes="table table-striped")
                return {
                    "table": html_output,
                    "format": "html", 
                    "row_count": len(table_data),
                    "columns": list(table_data[0].keys()) if table_data else []
                }
        
        # Default JSON format
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
    """Extract structured data from plain text"""
    lines = text.strip().split('\n')
    extracted_data = []
    
    # Simple extraction logic - can be enhanced
    for line in lines:
        if line.strip():
            if fields:
                # Try to map line content to fields
                parts = re.split(r'[,;:|]', line)
                row = {}
                for i, field in enumerate(fields):
                    if i < len(parts):
                        row[field] = parts[i].strip()
                    else:
                        row[field] = "N/A"
                extracted_data.append(row)
            else:
                # Simple key-value extraction
                if ':' in line:
                    key, value = line.split(':', 1)
                    extracted_data.append({
                        "key": key.strip(),
                        "value": value.strip()
                    })
                else:
                    extracted_data.append({"content": line.strip()})
    
    return extracted_data

if __name__ == "__main__":
    print("Starting Tabulator Server on port 9102...")
    # mcp.run(transport="sse", host="0.0.0.0", port=9102)
    mcp.run()

