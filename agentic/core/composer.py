# agentic/core/composer.py
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

from .types import StepResult
from .llm_client import llm_client

logger = logging.getLogger(__name__)

class ResultComposer:
    """Enhanced result composer with AI-powered analysis and formatting"""
    
    def __init__(self):
        self.composition_strategies = {
            "market_research": self._compose_market_research,
            "web_search": self._compose_web_search, 
            "data_analysis": self._compose_data_analysis,
            "calculation": self._compose_calculation,
            "sql_query": self._compose_sql_query,
            "workflow_execution": self._compose_workflow_execution,
            "system_query": self._compose_system_query,
            "default": self._compose_general
        }
    
    async def compose_results(self, results: List[StepResult], intent: str, 
                            style: str = "comprehensive") -> Dict[str, Any]:
        """Main composition entry point with AI enhancement"""
        try:
            if not results:
                return self._compose_empty_result(intent)
            
            # Separate successful and failed results
            successful_results = [r for r in results if r.success]
            failed_results = [r for r in results if not r.success]
            
            # Handle complete failure case
            if not successful_results:
                return self._compose_failure_result(failed_results, intent)
            
            # Determine composition strategy
            strategy = self._determine_composition_strategy(intent, successful_results)
            
            # Apply composition strategy
            composer_func = self.composition_strategies.get(strategy, self._compose_general)
            composed_result = composer_func(successful_results, failed_results, style)
            
            # FIXED: Make this function async and await properly
            if llm_client and llm_client.client and composed_result.get("result"):
                try:
                    enhanced_result = await self._enhance_with_ai(composed_result, intent, results)
                    composed_result.update(enhanced_result)
                except Exception as e:
                    logger.warning(f"AI enhancement failed: {e}")
            
            # Add composition metadata
            composed_result["composition_info"] = {
                "intent": intent,
                "strategy": strategy,
                "total_steps": len(results),
                "successful_steps": len(successful_results),
                "failed_steps": len(failed_results),
                "composition_timestamp": datetime.now().isoformat(),
                "style": style
            }
            
            return composed_result
            
        except Exception as e:
            logger.error(f"Result composition failed: {e}")
            return self._compose_error_result(str(e), results)
    
    def _determine_composition_strategy(self, intent: str, results: List[StepResult]) -> str:
        """Determine the best composition strategy based on intent and results"""
        
        # Intent-based strategy mapping
        intent_strategies = {
            "market_research": "market_research",
            "research": "market_research", 
            "search": "web_search",
            "web_search": "web_search",
            "analysis": "data_analysis",
            "data_analysis": "data_analysis",
            "calculate": "calculation",
            "calculation": "calculation",
            "sql": "sql_query",
            "database": "sql_query",
            "workflow": "workflow_execution",
            "system": "system_query"
        }
        
        # Check intent first
        for key, strategy in intent_strategies.items():
            if key.lower() in intent.lower():
                return strategy
        
        # Check agent patterns in results
        agent_patterns = [r.agent for r in results if r.agent]
        
        if any("web_search" in agent for agent in agent_patterns):
            return "web_search"
        elif any("tabulator" in agent for agent in agent_patterns):
            return "data_analysis"
        elif any("calculator" in agent for agent in agent_patterns):
            return "calculation"
        elif any("sql" in agent for agent in agent_patterns):
            return "sql_query"
        elif any("nlp" in agent for agent in agent_patterns):
            return "market_research"
        
        return "default"
    
    def _compose_market_research(self, successful_results: List[StepResult], 
                               failed_results: List[StepResult], 
                               style: str) -> Dict[str, Any]:
        """Compose market research results"""
        try:
            research_data = {
                "search_results": [],
                "summaries": [],
                "key_insights": [],
                "sources": [],
                "total_findings": 0
            }
            
            for result in successful_results:
                data = result.data
                
                if isinstance(data, dict):
                    # Web search results
                    if "results" in data and isinstance(data["results"], list):
                        research_data["search_results"].extend(data["results"])
                        research_data["total_findings"] += len(data["results"])
                    
                    # Summary data
                    if "summary" in data:
                        research_data["summaries"].append({
                            "source": result.agent,
                            "summary": data["summary"],
                            "key_points": data.get("key_points", [])
                        })
                    
                    # Insights
                    if "insights" in data:
                        research_data["key_insights"].extend(data["insights"])
                    
                    # Sources
                    if "sources" in data:
                        research_data["sources"].extend(data["sources"])
            
            # Deduplicate sources
            research_data["sources"] = list(set(research_data["sources"]))
            
            return {
                "type": "market_research",
                "result": research_data,
                "formatted_output": self._format_market_research(research_data, style)
            }
            
        except Exception as e:
            logger.error(f"Market research composition failed: {e}")
            return {"type": "market_research", "error": str(e)}
    
    def _compose_web_search(self, successful_results: List[StepResult], 
                          failed_results: List[StepResult], 
                          style: str) -> Dict[str, Any]:
        """Compose web search results"""
        try:
            search_data = {
                "total_results": 0,
                "results": [],
                "query_info": {},
                "sources": []
            }
            
            for result in successful_results:
                data = result.data
                
                if isinstance(data, dict) and "results" in data:
                    search_data["results"].extend(data["results"])
                    search_data["total_results"] += len(data["results"])
                    
                    if "query" in data:
                        search_data["query_info"][result.node_id] = data["query"]
                    
                    # Extract sources
                    for item in data["results"]:
                        if "url" in item:
                            search_data["sources"].append(item["url"])
            
            # Remove duplicates and limit results
            seen_urls = set()
            unique_results = []
            for item in search_data["results"]:
                url = item.get("url", "")
                if url not in seen_urls:
                    seen_urls.add(url)
                    unique_results.append(item)
            
            search_data["results"] = unique_results[:20]  # Limit to top 20
            search_data["total_results"] = len(unique_results)
            
            return {
                "type": "web_search",
                "result": search_data,
                "formatted_output": self._format_web_search(search_data, style)
            }
            
        except Exception as e:
            logger.error(f"Web search composition failed: {e}")
            return {"type": "web_search", "error": str(e)}
    
    def _compose_data_analysis(self, successful_results: List[StepResult], 
                             failed_results: List[StepResult], 
                             style: str) -> Dict[str, Any]:
        """Compose data analysis results"""
        try:
            analysis_data = {
                "tables": [],
                "charts": [],
                "metrics": {},
                "insights": [],
                "data_summary": {}
            }
            
            for result in successful_results:
                data = result.data
                
                if isinstance(data, dict):
                    # Tabulated data
                    if "table" in data:
                        analysis_data["tables"].append({
                            "source": result.agent,
                            "table": data["table"],
                            "metadata": data.get("metadata", {})
                        })
                    
                    # Metrics
                    if "metrics" in data:
                        analysis_data["metrics"].update(data["metrics"])
                    
                    # Charts/visualizations
                    if "chart" in data or "visualization" in data:
                        analysis_data["charts"].append(data)
                    
                    # Analysis insights
                    if "insights" in data:
                        analysis_data["insights"].extend(data["insights"])
                    
                    # Data summary
                    if "summary" in data:
                        analysis_data["data_summary"][result.node_id] = data["summary"]
            
            return {
                "type": "data_analysis",
                "result": analysis_data,
                "formatted_output": self._format_data_analysis(analysis_data, style)
            }
            
        except Exception as e:
            logger.error(f"Data analysis composition failed: {e}")
            return {"type": "data_analysis", "error": str(e)}
    
    def _compose_calculation(self, successful_results: List[StepResult], 
                           failed_results: List[StepResult], 
                           style: str) -> Dict[str, Any]:
        """Compose calculation results"""
        try:
            calc_data = {
                "calculations": [],
                "final_result": None,
                "formula": "",
                "steps": []
            }
            
            for result in successful_results:
                data = result.data
                
                if isinstance(data, dict):
                    if "result" in data:
                        calc_data["calculations"].append({
                            "step": result.node_id,
                            "calculation": data,
                            "result": data["result"]
                        })
                        calc_data["final_result"] = data["result"]  # Last result
                    
                    if "formula" in data:
                        calc_data["formula"] = data["formula"]
                    
                    if "steps" in data:
                        calc_data["steps"].extend(data["steps"])
            
            # Round decimal results to 3 places max
            if calc_data["final_result"] and isinstance(calc_data["final_result"], (int, float)):
                calc_data["final_result"] = round(calc_data["final_result"], 3)
            
            return {
                "type": "calculation",
                "result": calc_data,
                "formatted_output": self._format_calculation(calc_data, style)
            }
            
        except Exception as e:
            logger.error(f"Calculation composition failed: {e}")
            return {"type": "calculation", "error": str(e)}
    
    def _compose_sql_query(self, successful_results: List[StepResult], 
                         failed_results: List[StepResult], 
                         style: str) -> Dict[str, Any]:
        """Compose SQL query results"""
        try:
            sql_data = {
                "query_executed": True,
                "results": [],
                "total_rows": 0,
                "columns": [],
                "query_info": {},
                "execution_stats": {}
            }
            
            for result in successful_results:
                data = result.data
                
                if isinstance(data, dict):
                    if "data" in data:
                        sql_data["results"] = data["data"]
                        sql_data["total_rows"] = len(data["data"]) if isinstance(data["data"], list) else 0
                    
                    if "columns" in data:
                        sql_data["columns"] = data["columns"]
                    
                    if "query" in data:
                        sql_data["query_info"]["query"] = data["query"]
                    
                    if "execution_time" in data:
                        sql_data["execution_stats"]["execution_time"] = round(data["execution_time"], 3)
                    
                    if "affected_rows" in data:
                        sql_data["execution_stats"]["affected_rows"] = data["affected_rows"]
            
            return {
                "type": "sql_query",
                "result": sql_data,
                "formatted_output": self._format_sql_results(sql_data, style)
            }
            
        except Exception as e:
            logger.error(f"SQL query composition failed: {e}")
            return {"type": "sql_query", "error": str(e)}
    
    def _compose_workflow_execution(self, successful_results: List[StepResult], 
                                  failed_results: List[StepResult], 
                                  style: str) -> Dict[str, Any]:
        """Compose workflow execution results"""
        try:
            workflow_data = {
                "workflow_completed": True,
                "steps_executed": len(successful_results) + len(failed_results),
                "successful_steps": len(successful_results),
                "failed_steps": len(failed_results),
                "step_results": [],
                "final_output": "",
                "execution_summary": {}
            }
            
            # Compile step results
            all_results = successful_results + failed_results
            for result in all_results:
                workflow_data["step_results"].append({
                    "step": result.node_id,
                    "agent": result.agent,
                    "tool": result.tool,
                    "success": result.success,
                    "execution_time": round(result.execution_time, 3) if result.execution_time else 0,
                    "summary": self._summarize_step_result(result)
                })
            
            # Extract final output from last successful step
            if successful_results:
                last_result = successful_results[-1]
                workflow_data["final_output"] = self._extract_final_output(last_result.data)
            
            # Calculate execution summary
            total_time = sum(r.execution_time for r in all_results if r.execution_time)
            workflow_data["execution_summary"] = {
                "total_execution_time": round(total_time, 3),
                "success_rate": round(len(successful_results) / len(all_results) * 100, 1) if all_results else 0,
                "average_step_time": round(total_time / len(all_results), 3) if all_results else 0
            }
            
            return {
                "type": "workflow_execution",
                "result": workflow_data,
                "formatted_output": self._format_workflow_execution(workflow_data, style)
            }
            
        except Exception as e:
            logger.error(f"Workflow execution composition failed: {e}")
            return {"type": "workflow_execution", "error": str(e)}
    
    def _compose_system_query(self, successful_results: List[StepResult], 
                            failed_results: List[StepResult], 
                            style: str) -> Dict[str, Any]:
        """Compose system query results"""
        try:
            system_data = {
                "system_info": {},
                "data": [],
                "health_status": "healthy",
                "performance_metrics": {}
            }
            
            for result in successful_results:
                data = result.data
                
                if isinstance(data, dict):
                    if "data" in data:
                        system_data["data"] = data["data"]
                    else:
                        system_data["data"].append(data)
                
                system_data["system_info"][result.node_id] = {
                    "tool": result.tool,
                    "success": result.success,
                    "execution_time": round(result.execution_time, 3) if result.execution_time else 0
                }
            
            return {
                "type": "system_query",
                "result": system_data,
                "formatted_output": self._format_system_query(system_data, style)
            }
            
        except Exception as e:
            logger.error(f"System query composition failed: {e}")
            return {"type": "system_query", "error": str(e)}
    
    def _compose_general(self, successful_results: List[StepResult], 
                       failed_results: List[StepResult], 
                       style: str) -> Dict[str, Any]:
        """Compose general results when no specific strategy applies"""
        try:
            general_data = {
                "results": [],
                "summary": "",
                "combined_output": {},
                "agents_used": []
            }
            
            # Collect all successful results
            for result in successful_results:
                general_data["results"].append({
                    "step": result.node_id,
                    "agent": result.agent,
                    "tool": result.tool,
                    "data": result.data,
                    "execution_time": round(result.execution_time, 3) if result.execution_time else 0
                })
                
                if result.agent not in general_data["agents_used"]:
                    general_data["agents_used"].append(result.agent)
            
            # Try to extract meaningful output
            if successful_results:
                last_result = successful_results[-1]
                general_data["combined_output"] = last_result.data
                
                if isinstance(last_result.data, dict) and "summary" in last_result.data:
                    general_data["summary"] = last_result.data["summary"]
                elif len(successful_results) == 1:
                    general_data["summary"] = "Single step execution completed successfully"
                else:
                    general_data["summary"] = f"Multi-step workflow completed with {len(successful_results)} successful steps"
            
            return {
                "type": "general",
                "result": general_data,
                "formatted_output": self._format_general_results(general_data, style)
            }
            
        except Exception as e:
            logger.error(f"General composition failed: {e}")
            return {"type": "general", "error": str(e)}
    
    def _compose_empty_result(self, intent: str) -> Dict[str, Any]:
        """Compose result when no steps were executed"""
        return {
            "type": "empty",
            "result": {
                "message": "No steps were executed",
                "intent": intent,
                "suggestion": "Try rephrasing your query or check agent availability"
            },
            "formatted_output": "ℹ️ No workflow steps were executed. Please try a different query.",
            "composition_info": {
                "intent": intent,
                "total_steps": 0,
                "successful_steps": 0,
                "failed_steps": 0,
                "composition_timestamp": datetime.now().isoformat()
            }
        }
    
    def _compose_failure_result(self, results: List[StepResult], intent: str) -> Dict[str, Any]:
        """Compose result when all steps failed"""
        errors = [r.error for r in results if r.error]
        
        return {
            "type": "failure",
            "result": {
                "success": False,
                "total_steps": len(results),
                "failed_steps": len(results),
                "errors": errors,
                "intent": intent
            },
            "formatted_output": f"❌ Workflow failed: {len(results)} steps failed\n\nErrors:\n" + "\n".join(f"• {error}" for error in errors[:5]),
            "composition_info": {
                "intent": intent,
                "total_steps": len(results),
                "successful_steps": 0,
                "failed_steps": len(results),
                "composition_timestamp": datetime.now().isoformat()
            }
        }
    
    def _compose_error_result(self, error: str, results: List[StepResult]) -> Dict[str, Any]:
        """Compose result when composition itself fails"""
        return {
            "type": "error",
            "result": {
                "composition_error": error,
                "total_steps": len(results),
                "raw_results": [
                    {
                        "node_id": r.node_id,
                        "success": r.success,
                        "data": r.data,
                        "error": r.error
                    }
                    for r in results
                ]
            },
            "formatted_output": f"❌ Result composition failed: {error}",
            "composition_info": {
                "composition_error": error,
                "total_steps": len(results),
                "composition_timestamp": datetime.now().isoformat()
            }
        }
    
    # AI Enhancement - FIXED: Now properly async
    async def _enhance_with_ai(self, composed_result: Dict[str, Any], intent: str, 
                             results: List[StepResult]) -> Dict[str, Any]:
        """Enhance composed results with AI analysis"""
        try:
            if not composed_result.get("result"):
                return {}
            
            enhancement_prompt = f"""
            Enhance this workflow result for intent '{intent}':
            
            Composed Result:
            {json.dumps(composed_result["result"], indent=2)}
            
            Provide enhancements including:
            1. Key insights and takeaways
            2. Executive summary (2-3 sentences)
            3. Action items or recommendations
            4. Quality assessment of results
            5. Confidence score (0-100)
            
            Return as JSON with keys: insights, executive_summary, recommendations, quality_assessment, confidence_score
            """
            
            messages = [
                {"role": "system", "content": "You are an expert result analyzer. Provide structured insights and recommendations based on workflow results."},
                {"role": "user", "content": enhancement_prompt}
            ]
            
            response = await llm_client.generate(messages, temperature=0.3, max_tokens=800)
            if response:
                try:
                    enhancement = json.loads(response.strip())
                    return {
                        "ai_enhancement": enhancement,
                        "enhanced_summary": enhancement.get("executive_summary", ""),
                        "confidence_score": enhancement.get("confidence_score", 75)
                    }
                except json.JSONDecodeError:
                    logger.warning("Failed to parse AI enhancement response")
                    return {
                        "ai_enhancement": {"raw_response": response},
                        "enhanced_summary": response[:200] + "..." if len(response) > 200 else response
                    }
            
            return {}
            
        except Exception as e:
            logger.error(f"AI enhancement failed: {e}")
            return {}
    
    # Formatting methods
    def _format_market_research(self, data: Dict[str, Any], style: str) -> str:
        """Format market research results"""
        if style == "brief":
            return f"📊 Market Research Summary\n\nFound {data['total_findings']} results across {len(data['summaries'])} sources.\n\nKey insights: {', '.join(data['key_insights'][:3])}"
        else:
            output = "# 📊 Market Research Report\n\n"
            output += f"## Executive Summary\nAnalyzed {data['total_findings']} findings from {len(data['sources'])} sources.\n\n"
            
            if data['summaries']:
                output += "## Key Findings\n"
                for summary in data['summaries'][:3]:
                    output += f"- **{summary['source']}**: {summary['summary']}\n"
            
            if data['key_insights']:
                output += "\n## Key Insights\n"
                for insight in data['key_insights'][:5]:
                    output += f"• {insight}\n"
            
            return output
    
    def _format_web_search(self, data: Dict[str, Any], style: str) -> str:
        """Format web search results"""
        if style == "brief":
            return f"🔍 Found {data['total_results']} search results"
        else:
            output = f"# 🔍 Search Results ({data['total_results']} found)\n\n"
            for i, result in enumerate(data['results'][:5], 1):
                title = result.get('title', 'No title')
                snippet = result.get('snippet', result.get('description', 'No description'))
                url = result.get('url', '#')
                output += f"## {i}. {title}\n{snippet}\n[Read more]({url})\n\n"
            return output
    
    def _format_data_analysis(self, data: Dict[str, Any], style: str) -> str:
        """Format data analysis results"""
        if style == "brief":
            return f"📈 Analysis complete: {len(data['tables'])} tables, {len(data['insights'])} insights"
        else:
            output = "# 📈 Data Analysis Report\n\n"
            
            if data['metrics']:
                output += "## Key Metrics\n"
                for key, value in data['metrics'].items():
                    if isinstance(value, (int, float)):
                        value = round(value, 3) if isinstance(value, float) else value
                    output += f"- **{key}**: {value}\n"
                output += "\n"
            
            if data['insights']:
                output += "## Insights\n"
                for insight in data['insights'][:5]:
                    output += f"• {insight}\n"
            
            return output
    
    def _format_calculation(self, data: Dict[str, Any], style: str) -> str:
        """Format calculation results"""
        result = data.get('final_result', 'No result')
        if isinstance(result, (int, float)):
            result = round(result, 3) if isinstance(result, float) else result
        
        if style == "brief":
            return f"🔢 Calculation result: {result}"
        else:
            output = "# 🔢 Calculation Results\n\n"
            output += f"**Final Result**: {result}\n\n"
            
            if data.get('formula'):
                output += f"**Formula**: {data['formula']}\n\n"
            
            if data.get('steps'):
                output += "## Calculation Steps\n"
                for i, step in enumerate(data['steps'], 1):
                    output += f"{i}. {step}\n"
            
            return output
    
    def _format_sql_results(self, data: Dict[str, Any], style: str) -> str:
        """Format SQL query results"""
        if style == "brief":
            return f"💾 Query executed: {data['total_rows']} rows returned"
        else:
            output = "# 💾 SQL Query Results\n\n"
            
            if data.get('query_info', {}).get('query'):
                output += f"**Query**: `{data['query_info']['query']}`\n\n"
            
            output += f"**Rows returned**: {data['total_rows']}\n"
            
            if data.get('execution_stats', {}).get('execution_time'):
                output += f"**Execution time**: {data['execution_stats']['execution_time']}s\n"
            
            if data['results'] and len(data['results']) > 0:
                output += "\n## Sample Results\n"
                sample = data['results'][:3]
                for i, row in enumerate(sample, 1):
                    output += f"Row {i}: {row}\n"
            
            return output
    
    def _format_workflow_execution(self, data: Dict[str, Any], style: str) -> str:
        """Format workflow execution results"""
        if style == "brief":
            return f"⚡ Workflow completed: {data['successful_steps']}/{data['steps_executed']} steps successful"
        else:
            output = "# ⚡ Workflow Execution Report\n\n"
            output += f"**Success Rate**: {data['execution_summary']['success_rate']}%\n"
            output += f"**Total Time**: {data['execution_summary']['total_execution_time']}s\n"
            output += f"**Steps**: {data['successful_steps']}/{data['steps_executed']} successful\n\n"
            
            if data['final_output']:
                output += f"## Final Output\n{data['final_output']}\n\n"
            
            output += "## Step Summary\n"
            for step in data['step_results']:
                status = "✅" if step['success'] else "❌"
                output += f"{status} **{step['step']}** ({step['agent']}.{step['tool']}) - {step['execution_time']}s\n"
            
            return output
    
    def _format_system_query(self, data: Dict[str, Any], style: str) -> str:
        """Format system query results"""
        if style == "brief":
            return f"🖥️ System query completed: {len(data['data'])} records"
        else:
            output = "# 🖥️ System Query Results\n\n"
            output += f"**Status**: {data['health_status']}\n"
            output += f"**Records**: {len(data['data'])}\n\n"
            
            if data['system_info']:
                output += "## System Info\n"
                for key, info in data['system_info'].items():
                    output += f"- **{key}**: {info['tool']} ({info['execution_time']}s)\n"
            
            return output
    
    def _format_general_results(self, data: Dict[str, Any], style: str) -> str:
        """Format general results"""
        if style == "brief":
            return f"✅ {len(data['results'])} steps completed using {len(data['agents_used'])} agents"
        else:
            output = "# ✅ Workflow Results\n\n"
            output += f"**Summary**: {data['summary']}\n"
            output += f"**Agents Used**: {', '.join(data['agents_used'])}\n\n"
            
            if data['results']:
                output += "## Step Results\n"
                for result in data['results'][:5]:
                    output += f"- **{result['step']}** ({result['agent']}.{result['tool']}) - {result['execution_time']}s\n"
            
            return output
    
    # Helper methods
    def _summarize_step_result(self, result: StepResult) -> str:
        """Create a brief summary of a step result"""
        if not result.success:
            return f"Failed: {result.error}"
        
        if isinstance(result.data, dict):
            if "summary" in result.data:
                return result.data["summary"]
            elif "results" in result.data:
                count = len(result.data["results"]) if isinstance(result.data["results"], list) else 1
                return f"Returned {count} results"
            elif "data" in result.data:
                count = len(result.data["data"]) if isinstance(result.data["data"], list) else 1
                return f"Processed {count} records"
        
        return "Completed successfully"
    
    def _extract_final_output(self, data: Any) -> str:
        """Extract the most relevant final output from step data"""
        if isinstance(data, dict):
            # Priority order for output extraction
            for key in ["summary", "result", "formatted_output", "data"]:
                if key in data:
                    value = data[key]
                    if isinstance(value, str) and len(value) > 0:
                        return value[:500]  # Limit length
                    elif isinstance(value, (list, dict)):
                        return str(value)[:500]
        
        return str(data)[:500] if data else "No output available"

# Create global instance
composer = ResultComposer()