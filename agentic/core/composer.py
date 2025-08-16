# ============================================================================
# agentic/core/composer.py
import json
from typing import List, Dict, Any, Optional
import logging
from .types import StepResult

logger = logging.getLogger(__name__)

class ResultComposer:
    def compose_results(self, results: List[StepResult], intent: str) -> Dict[str, Any]:
        """Compose final result from step results"""
        if not results:
            return {"error": "No results to compose"}
        
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        if intent == "market_research":
            return self._compose_market_research(successful_results, failed_results)
        elif intent == "web_search":
            return self._compose_search_results(successful_results, failed_results)
        elif intent == "data_analysis":
            return self._compose_analysis_results(successful_results, failed_results)
        else:
            return self._compose_default_results(successful_results, failed_results)
    
    def _compose_market_research(self, successful: List[StepResult], failed: List[StepResult]) -> Dict[str, Any]:
        """Compose market research results"""
        findings = []
        sources = []
        
        for result in successful:
            if result.tool == "search":
                try:
                    data = json.loads(result.data) if isinstance(result.data, str) else result.data
                    if isinstance(data, dict) and "items" in data:
                        for item in data["items"]:
                            findings.append({
                                "title": item.get("title", ""),
                                "snippet": item.get("snippet", ""),
                                "url": item.get("url", "")
                            })
                            sources.append(item.get("url", ""))
                except:
                    findings.append({"raw_data": result.data})
            elif result.tool == "summarize":
                findings.append({"summary": result.data})
            elif result.tool == "tabulate":
                findings.append({"table": result.data})
        
        return {
            "type": "market_research",
            "findings": findings,
            "sources": list(set(sources)),
            "total_results": len(successful),
            "failed_steps": len(failed),
            "success": len(failed) == 0
        }
    
    def _compose_search_results(self, successful: List[StepResult], failed: List[StepResult]) -> Dict[str, Any]:
        """Compose search results"""
        all_items = []
        
        for result in successful:
            try:
                data = json.loads(result.data) if isinstance(result.data, str) else result.data
                if isinstance(data, dict) and "items" in data:
                    all_items.extend(data["items"])
                else:
                    all_items.append({"data": result.data})
            except:
                all_items.append({"raw_data": result.data})
        
        return {
            "type": "search_results",
            "items": all_items,
            "total_items": len(all_items),
            "failed_steps": len(failed),
            "success": len(failed) == 0
        }
    
    def _compose_analysis_results(self, successful: List[StepResult], failed: List[StepResult]) -> Dict[str, Any]:
        """Compose analysis results"""
        analysis = {}
        tables = []
        
        for result in successful:
            if result.tool == "analyze":
                analysis["analysis"] = result.data
            elif result.tool == "tabulate":
                tables.append(result.data)
            else:
                analysis[result.tool] = result.data
        
        return {
            "type": "analysis_results",
            "analysis": analysis,
            "tables": tables,
            "failed_steps": len(failed),
            "success": len(failed) == 0
        }
    
    def _compose_default_results(self, successful: List[StepResult], failed: List[StepResult]) -> Dict[str, Any]:
        """Default result composition"""
        results_by_tool = {}
        
        for result in successful:
            if result.tool not in results_by_tool:
                results_by_tool[result.tool] = []
            results_by_tool[result.tool].append(result.data)
        
        return {
            "type": "general_results",
            "results": results_by_tool,
            "total_steps": len(successful),
            "failed_steps": len(failed),
            "success": len(failed) == 0
        }

composer = ResultComposer()