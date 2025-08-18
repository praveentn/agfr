# agentic/agents/local/tabulator_server.py
import json
import re
import csv
import io
from typing import Dict, List, Any, Union
import logging

import pandas as pd
from fastmcp import FastMCP

mcp = FastMCP("Tabulator Server")

@mcp.tool()
def tabulate(data: Union[str, List, Dict], fields: List[str] = None, format: str = "json") -> Dict[str, Any]:
    """
    Convert input data into structured table format (JSON, CSV, or HTML).
    Handles various input formats: text, JSON, search results, summaries.
    """
    try:
        print(f"Tabulating data: type={type(data)}, format={format}")
        
        # Parse and normalize input data
        if isinstance(data, str):
            try:
                # Try to parse as JSON first
                parsed = json.loads(data)
                print("Parsed data as JSON")
            except json.JSONDecodeError:
                # Handle text data
                parsed = _extract_data_from_text(data, fields)
                print(f"Extracted {len(parsed)} items from text")
        else:
            parsed = data

        # Normalize to list of dictionaries
        table_data = _normalize_to_table(parsed)
        print(f"Normalized to {len(table_data)} table rows")

        # Filter and organize fields
        if fields:
            table_data = _filter_fields(table_data, fields)
        else:
            # Auto-detect common fields if not specified
            fields = _auto_detect_fields(table_data)

        # Apply format-specific processing
        if format.lower() == "csv":
            return _format_as_csv(table_data, fields)
        elif format.lower() == "html":
            return _format_as_html(table_data, fields)
        else:
            return _format_as_json(table_data, fields)

    except Exception as e:
        print(f"Tabulation error: {e}")
        return {
            "table": [],
            "error": str(e),
            "format": format,
            "row_count": 0,
            "columns": []
        }

def _normalize_to_table(data: Any) -> List[Dict]:
    """Normalize various data formats to list of dictionaries"""
    if isinstance(data, dict):
        # Handle search results format
        if "items" in data:
            return data["items"]
        elif "findings" in data:
            return data["findings"]
        elif "summary" in data and "key_points" in data:
            # Handle summarization results
            return [{
                "type": "summary",
                "content": data["summary"],
                "key_points": ", ".join(data.get("key_points", [])),
                "word_count": data.get("word_count", 0)
            }]
        else:
            # Single dictionary
            return [data]
    elif isinstance(data, list):
        # List of items
        normalized = []
        for item in data:
            if isinstance(item, dict):
                normalized.append(item)
            else:
                normalized.append({"value": str(item)})
        return normalized
    else:
        # Single value
        return [{"value": str(data)}]

def _extract_data_from_text(text: str, fields: List[str] = None) -> List[Dict]:
    """Extract structured data from plain text"""
    rows = []
    lines = text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Try different parsing strategies
        if fields:
            # Parse with expected fields
            if any(delim in line for delim in [',', ';', '|', '\t']):
                # Delimited data
                for delim in [',', ';', '|', '\t']:
                    if delim in line:
                        parts = [p.strip() for p in line.split(delim)]
                        row = {}
                        for i, field in enumerate(fields):
                            row[field] = parts[i] if i < len(parts) else "N/A"
                        rows.append(row)
                        break
            elif ':' in line:
                # Key-value pairs
                key, value = line.split(':', 1)
                if len(fields) >= 2:
                    rows.append({fields[0]: key.strip(), fields[1]: value.strip()})
                else:
                    rows.append({"key": key.strip(), "value": value.strip()})
            else:
                # Single field content
                if fields:
                    rows.append({fields[0]: line})
                else:
                    rows.append({"content": line})
        else:
            # Auto-detect structure
            if ':' in line and len(line.split(':')) == 2:
                key, value = line.split(':', 1)
                rows.append({"key": key.strip(), "value": value.strip()})
            elif any(delim in line for delim in [',', ';', '|']):
                # Try to parse delimited data
                for delim in [',', ';', '|']:
                    if delim in line:
                        parts = [p.strip() for p in line.split(delim)]
                        if len(parts) >= 2:
                            row = {f"column_{i+1}": part for i, part in enumerate(parts)}
                            rows.append(row)
                        break
            else:
                rows.append({"content": line})
    
    return rows

def _auto_detect_fields(table_data: List[Dict]) -> List[str]:
    """Auto-detect important fields from data"""
    if not table_data:
        return []
    
    # Collect all unique keys
    all_keys = set()
    for row in table_data:
        all_keys.update(row.keys())
    
    # Prioritize common fields
    priority_fields = [
        "title", "name", "subject", "topic",
        "url", "link", "source", 
        "snippet", "description", "summary", "content",
        "date", "date_published", "timestamp",
        "price", "value", "amount", "score",
        "type", "category", "status"
    ]
    
    ordered_fields = []
    
    # Add priority fields first
    for field in priority_fields:
        if field in all_keys:
            ordered_fields.append(field)
            all_keys.remove(field)
    
    # Add remaining fields
    ordered_fields.extend(sorted(all_keys))
    
    return ordered_fields

def _filter_fields(table_data: List[Dict], fields: List[str]) -> List[Dict]:
    """Filter data to only include specified fields"""
    filtered_data = []
    for row in table_data:
        filtered_row = {}
        for field in fields:
            value = row.get(field, "N/A")
            # Clean up the value
            if isinstance(value, str):
                value = value.strip()
                if len(value) > 200:  # Truncate very long text
                    value = value[:197] + "..."
            filtered_row[field] = value
        filtered_data.append(filtered_row)
    
    return filtered_data

def _format_as_json(table_data: List[Dict], fields: List[str]) -> Dict[str, Any]:
    """Format data as JSON"""
    return {
        "table": table_data,
        "format": "json",
        "row_count": len(table_data),
        "columns": fields or (list(table_data[0].keys()) if table_data else [])
    }

def _format_as_csv(table_data: List[Dict], fields: List[str]) -> Dict[str, Any]:
    """Format data as CSV string"""
    if not table_data:
        return {
            "table": "",
            "format": "csv", 
            "row_count": 0,
            "columns": []
        }
    
    # Use pandas for clean CSV generation
    df = pd.DataFrame(table_data)
    if fields:
        # Reorder columns and ensure all fields exist
        existing_fields = [f for f in fields if f in df.columns]
        missing_fields = [f for f in fields if f not in df.columns]
        
        # Add missing fields with N/A
        for field in missing_fields:
            df[field] = "N/A"
        
        df = df[fields]
    
    csv_output = df.to_csv(index=False)
    
    return {
        "table": csv_output,
        "format": "csv",
        "row_count": len(table_data),
        "columns": list(df.columns)
    }

def _format_as_html(table_data: List[Dict], fields: List[str]) -> Dict[str, Any]:
    """Format data as HTML table"""
    if not table_data:
        return {
            "table": "<p>No data to display</p>",
            "format": "html",
            "row_count": 0,
            "columns": []
        }
    
    df = pd.DataFrame(table_data)
    if fields:
        # Reorder columns and ensure all fields exist
        existing_fields = [f for f in fields if f in df.columns]
        missing_fields = [f for f in fields if f not in df.columns]
        
        # Add missing fields with N/A
        for field in missing_fields:
            df[field] = "N/A"
        
        df = df[fields]
    
    # Generate clean HTML table
    html_output = df.to_html(
        index=False, 
        classes="table table-striped table-hover",
        escape=False,
        border=0
    )
    
    return {
        "table": html_output,
        "format": "html", 
        "row_count": len(table_data),
        "columns": list(df.columns)
    }

@mcp.tool()
def sort_data(data: Union[str, List[Dict]], sort_field: str, ascending: bool = True) -> Dict[str, Any]:
    """Sort tabular data by specified field."""
    try:
        # Parse data
        if isinstance(data, str):
            parsed_data = json.loads(data)
        else:
            parsed_data = data
        
        table_data = _normalize_to_table(parsed_data)
        
        if not table_data:
            return {"table": [], "row_count": 0, "columns": []}
        
        # Sort data
        if sort_field in table_data[0]:
            sorted_data = sorted(
                table_data, 
                key=lambda x: x.get(sort_field, ""), 
                reverse=not ascending
            )
        else:
            sorted_data = table_data
        
        return {
            "table": sorted_data,
            "format": "json",
            "row_count": len(sorted_data),
            "columns": list(sorted_data[0].keys()) if sorted_data else [],
            "sorted_by": sort_field,
            "ascending": ascending
        }
        
    except Exception as e:
        return {
            "table": [],
            "error": str(e),
            "format": "json",
            "row_count": 0
        }

if __name__ == "__main__":
    print("Starting Tabulator Server on port 9102...")
    
    # Use sync run to avoid asyncio conflicts
    mcp.run(transport="http", host="0.0.0.0", port=9102)