# modules/search_engine.py
import requests
import time
import json
import re
import os
import glob
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.common.exceptions import TimeoutException, WebDriverException
from bs4 import BeautifulSoup
import sys
import ssl

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SEARCH_CONFIG, SCRAPING_CONFIG, get_error_message, logger
from utils import sanitize_url, extract_domain, rate_limit

class WebSearchEngine:
    """Comprehensive web search engine with multiple search providers, Selenium support, and local file fallback"""
    
    def __init__(self):
        self.config = SEARCH_CONFIG
        self.scraping_config = SCRAPING_CONFIG
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.scraping_config['user_agent']
        })
        # Disable SSL verification for scraping fallback
        self.session.verify = False
        self.driver = None
        self._init_selenium()
        self.local_data_path = os.path.join(os.path.dirname(__file__), '..', 'local_data')
        if not os.path.exists(self.local_data_path):
            os.makedirs(self.local_data_path)

    def _init_selenium(self):
        """Initialize Selenium WebDriver with improved error handling"""
        if not self.config.get('use_selenium', True):
            return
        
        try:
            # Try Chrome first, then Edge
            try:
                chrome_options = ChromeOptions()
                if self.config.get('selenium_headless', True):
                    chrome_options.add_argument('--headless')
                chrome_options.add_argument('--no-sandbox')
                chrome_options.add_argument('--disable-dev-shm-usage')
                chrome_options.add_argument('--disable-gpu')
                chrome_options.add_argument('--window-size=1920,1080')
                chrome_options.add_argument('--disable-web-security')
                chrome_options.add_argument('--allow-running-insecure-content')
                chrome_options.add_argument('--disable-extensions')
                chrome_options.add_argument('--disable-plugins')
                chrome_options.add_argument('--disable-images')
                chrome_options.add_argument(f"--user-agent={self.scraping_config['user_agent']}")
                
                from webdriver_manager.chrome import ChromeDriverManager
                from selenium.webdriver.chrome.service import Service
                service = Service(ChromeDriverManager().install())
                self.driver = webdriver.Chrome(service=service, options=chrome_options)
                self.driver.set_page_load_timeout(30)
                logger.info("Chrome WebDriver initialized successfully")
            except Exception as e:
                logger.warning(f"Chrome WebDriver failed: {e}, trying Edge...")
                
                edge_options = EdgeOptions()
                if self.config.get('selenium_headless', True):
                    edge_options.add_argument('--headless')
                edge_options.add_argument('--no-sandbox')
                edge_options.add_argument('--disable-dev-shm-usage')
                edge_options.add_argument('--disable-gpu')
                edge_options.add_argument('--window-size=1920,1080')
                edge_options.add_argument('--disable-web-security')
                edge_options.add_argument('--allow-running-insecure-content')
                edge_options.add_argument(f"--user-agent={self.scraping_config['user_agent']}")
                
                from webdriver_manager.microsoft import EdgeChromiumDriverManager
                from selenium.webdriver.edge.service import Service as EdgeService
                service = EdgeService(EdgeChromiumDriverManager().install())
                self.driver = webdriver.Edge(service=service, options=edge_options)
                self.driver.set_page_load_timeout(30)
                logger.info("Edge WebDriver initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize WebDriver: {e}")
            self.driver = None

    def comprehensive_search(self, query: str, search_type: str = "general", max_results: int = 10) -> List[Dict[str, Any]]:
        """Perform comprehensive search using multiple sources with local file fallback"""
        all_results = []
        
        # Generate search queries based on the type
        search_queries = self._generate_search_queries(query, search_type)
        
        # First try API-based searches
        for search_query in search_queries[:3]:  # Limit to 3 different query variations
            try:
                # Try different search methods
                results = []
                
                # Method 1: Bing Search (if API key available)
                if self.config.get('bing_api_key'):
                    try:
                        bing_results = self._search_bing(search_query, max_results//3)
                        results.extend(bing_results)
                    except Exception as e:
                        logger.warning(f"Bing search failed: {e}")
                
                # Method 2: Google Custom Search (if API key available)
                if self.config.get('google_api_key') and self.config.get('google_cse_id'):
                    try:
                        google_results = self._search_google_custom(search_query, max_results//3)
                        results.extend(google_results)
                    except Exception as e:
                        logger.warning(f"Google search failed: {e}")
                
                # Method 3: SerpAPI (if API key available)
                if self.config.get('serpapi_key'):
                    try:
                        serp_results = self._search_serpapi(search_query, max_results//3)
                        results.extend(serp_results)
                    except Exception as e:
                        logger.warning(f"SerpAPI search failed: {e}")
                
                # Method 4: Selenium-based search (fallback) - Disabled due to repeated failures
                # if not results and self.driver:
                #     try:
                #         selenium_results = self._search_with_selenium(search_query, max_results//2)
                #         results.extend(selenium_results)
                #     except Exception as e:
                #         logger.warning(f"Selenium search failed: {e}")
                
                # Method 5: Direct web scraping (with SSL error handling)
                if not results:
                    try:
                        scraping_results = self._search_with_scraping(search_query, max_results//2)
                        results.extend(scraping_results)
                    except Exception as e:
                        logger.warning(f"Scraping search failed: {e}")
                
                all_results.extend(results)
                
                # Add delay between searches
                time.sleep(self.scraping_config.get('request_delay', 1.0))
                
            except Exception as e:
                logger.error(f"Search failed for query '{search_query}': {e}")
                continue
        
        # If no results from web search, try local file search
        if not all_results:
            logger.info("No web search results found, trying local file search...")
            local_results = self._search_local_files(query, search_type, max_results)
            all_results.extend(local_results)
        
        # If still no results, create minimal mock results to prevent total failure
        if not all_results:
            logger.warning("No search results found, creating fallback results")
            all_results = self._create_fallback_results(query, search_type)
        
        # Remove duplicates and enhance results
        unique_results = self._deduplicate_results(all_results)
        enhanced_results = self._enhance_search_results(unique_results[:max_results])
        
        logger.info(f"Comprehensive search returned {len(enhanced_results)} results for query: {query}")
        return enhanced_results

    def _search_local_files(self, query: str, search_type: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for local text files with names matching the search term"""
        results = []
        
        try:
            # Clean query for filename matching
            clean_query = re.sub(r'[^\w\s-]', '', query.lower())
            query_words = clean_query.split()
            
            # Search patterns
            search_patterns = [
                f"*{clean_query}*",
                f"*{query_words[0]}*" if query_words else "",
            ]
            
            # Add type-specific patterns
            if search_type == "company":
                search_patterns.extend([f"*{clean_query}_company*", f"*{clean_query}_analysis*"])
            elif search_type == "startup":
                search_patterns.extend([f"*{clean_query}_startup*", f"*{clean_query}_funding*"])
            
            # Search for text files
            file_extensions = ['*.txt', '*.md', '*.json', '*.csv']
            
            for pattern in search_patterns:
                if not pattern:
                    continue
                    
                for ext in file_extensions:
                    search_path = os.path.join(self.local_data_path, f"{pattern}.{ext.replace('*.', '')}")
                    matching_files = glob.glob(search_path, recursive=True)
                    
                    for file_path in matching_files[:max_results]:
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                                
                            # Create result object
                            result = {
                                "title": f"Local Analysis: {os.path.basename(file_path)}",
                                "url": f"file://{file_path}",
                                "snippet": content[:300] + "..." if len(content) > 300 else content,
                                "full_content": content[:2000] if len(content) > 2000 else content,
                                "domain": "local_file",
                                "source": "local_file",
                                "relevance_score": 0.8,
                                "trusted_source": True,
                                "date_published": time.ctime(os.path.getmtime(file_path)),
                                "content_richness": 0.9
                            }
                            results.append(result)
                            
                        except Exception as e:
                            logger.debug(f"Error reading local file {file_path}: {e}")
                            continue
            
            logger.info(f"Local file search found {len(results)} results for query: {query}")
            return results[:max_results]
            
        except Exception as e:
            logger.error(f"Local file search failed: {e}")
            return []

    def _create_fallback_results(self, query: str, search_type: str) -> List[Dict[str, Any]]:
        """Create minimal fallback results when all search methods fail"""
        fallback_results = []
        
        # Create general analysis guidance
        general_guidance = {
            "company": f"Analysis of {query} requires examining market position, competitive landscape, technology adoption, and industry disruption trends.",
            "startup": f"Startup analysis for {query} should focus on business model innovation, market opportunity, competitive advantages, and scaling potential.",
            "sector": f"Sector analysis for {query} involves studying market trends, regulatory changes, technological disruptions, and emerging business models.",
            "geography": f"Geographic analysis for {query} requires understanding economic conditions, innovation ecosystems, regulatory environment, and market opportunities.",
            "technology": f"Technology analysis for {query} should examine adoption trends, market applications, competitive landscape, and disruption potential."
        }
        
        guidance_text = general_guidance.get(search_type, f"Analysis of {query} requires comprehensive market research and trend analysis.")
        
        fallback_result = {
            "title": f"Analysis Framework for {query}",
            "url": "internal://analysis_framework",
            "snippet": guidance_text,
            "full_content": guidance_text + " " + self._get_analysis_framework(search_type),
            "domain": "internal",
            "source": "fallback",
            "relevance_score": 0.6,
            "trusted_source": True,
            "date_published": time.strftime('%Y-%m-%d'),
            "content_richness": 0.7
        }
        
        fallback_results.append(fallback_result)
        return fallback_results

    def _get_analysis_framework(self, search_type: str) -> str:
        """Get analysis framework text for fallback results"""
        frameworks = {
            "company": "Key analysis areas: 1) Financial performance and market share, 2) Product innovation and R&D investment, 3) Competitive positioning, 4) Technology adoption, 5) Market expansion strategies, 6) Risk factors and regulatory challenges.",
            "startup": "Startup evaluation framework: 1) Market opportunity size and growth, 2) Business model scalability, 3) Team expertise and track record, 4) Product-market fit validation, 5) Competitive differentiation, 6) Funding strategy and financial projections.",
            "sector": "Industry analysis components: 1) Market size and growth trends, 2) Competitive landscape mapping, 3) Technology disruption factors, 4) Regulatory environment changes, 5) Consumer behavior shifts, 6) Investment and M&A activity.",
            "geography": "Geographic analysis elements: 1) Economic indicators and growth outlook, 2) Innovation ecosystem maturity, 3) Regulatory and policy environment, 4) Infrastructure and talent availability, 5) Market accessibility and barriers, 6) Cultural and social factors.",
            "technology": "Technology assessment framework: 1) Technical maturity and readiness, 2) Market adoption curve and timeline, 3) Competitive technology landscape, 4) Application areas and use cases, 5) Investment and funding trends, 6) Regulatory and ethical considerations."
        }
        return frameworks.get(search_type, "Comprehensive analysis requires systematic evaluation of market, competitive, technological, and regulatory factors.")

    def _search_with_scraping(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search using direct web scraping with improved SSL handling"""
        try:
            # Create unverified HTTPS context
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # Use DuckDuckGo as it's more scraping-friendly
            search_url = f"https://html.duckduckgo.com/html/?q={query}"
            
            # Create session with SSL verification disabled
            session = requests.Session()
            session.verify = False
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            # Disable SSL warnings
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            
            response = session.get(search_url, timeout=15, verify=False)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            results = []
            
            # Extract search results with multiple selectors
            result_selectors = [
                'div.web-result',
                'div.result',
                'div.results_links',
                'div.result__body'
            ]
            
            result_elements = []
            for selector in result_selectors:
                elements = soup.select(selector)
                if elements:
                    result_elements = elements[:max_results]
                    break
            
            for element in result_elements:
                try:
                    # Try multiple selector patterns
                    title_elem = (element.select_one('h2.result__title a') or 
                                element.select_one('.result__title') or
                                element.select_one('h3') or
                                element.select_one('h2'))
                    
                    link_elem = (element.select_one('a.result__a') or
                               element.select_one('a[href]'))
                    
                    snippet_elem = (element.select_one('.result__snippet') or
                                  element.select_one('.snippet') or
                                  element.select_one('p'))
                    
                    if title_elem and link_elem:
                        title = title_elem.get_text(strip=True)
                        url = link_elem.get('href', '')
                        snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                        
                        # Clean URL if it's a DuckDuckGo redirect
                        if url.startswith('/l/?kh=-1&uddg='):
                            url = url.split('&uddg=')[-1] if '&uddg=' in url else url
                        
                        result = {
                            "title": title,
                            "url": url,
                            "snippet": snippet,
                            "date_published": "",
                            "source": "scraping",
                            "relevance_score": 0.6,
                            "domain": extract_domain(url) if url else "unknown",
                            "trusted_source": True,
                            "content_richness": 0.5
                        }
                        results.append(result)
                        
                except Exception as e:
                    logger.debug(f"Error extracting scraped result: {e}")
                    continue
            
            logger.info(f"Scraping search returned {len(results)} results for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Scraping search failed: {e}")
            return []

    # Keep all the existing rate-limited API methods unchanged
    @rate_limit(calls=10, period=60)
    def _search_bing(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search using Bing Search API"""
        if not self.config.get('bing_api_key'):
            return []
        
        try:
            endpoint = "https://api.bing.microsoft.com/v7.0/search"
            headers = {"Ocp-Apim-Subscription-Key": self.config['bing_api_key']}
            params = {
                "q": query,
                "count": min(max_results, 50),
                "textDecorations": True,
                "textFormat": "HTML",
                "freshness": "Month"
            }
            
            response = self.session.get(endpoint, headers=headers, params=params, 
                                      timeout=self.config.get('search_timeout', 30))
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get("webPages", {}).get("value", [])[:max_results]:
                result = {
                    "title": item.get("name", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("snippet", ""),
                    "date_published": item.get("dateLastCrawled", ""),
                    "source": "bing",
                    "relevance_score": 0.8,
                    "domain": extract_domain(item.get("url", "")),
                    "trusted_source": True,
                    "content_richness": 0.8
                }
                results.append(result)
            
            logger.info(f"Bing search returned {len(results)} results for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Bing search failed: {e}")
            return []

    @rate_limit(calls=100, period=86400)
    def _search_google_custom(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search using Google Custom Search API"""
        if not self.config.get('google_api_key') or not self.config.get('google_cse_id'):
            return []
        
        try:
            endpoint = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.config['google_api_key'],
                "cx": self.config['google_cse_id'],
                "q": query,
                "num": min(max_results, 10),
                "dateRestrict": "m6"
            }
            
            response = self.session.get(endpoint, params=params, 
                                      timeout=self.config.get('search_timeout', 30))
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get("items", [])[:max_results]:
                result = {
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "date_published": item.get("pagemap", {}).get("metatags", [{}])[0].get("article:published_time", ""),
                    "source": "google",
                    "relevance_score": 0.9,
                    "domain": extract_domain(item.get("link", "")),
                    "trusted_source": True,
                    "content_richness": 0.9
                }
                results.append(result)
            
            logger.info(f"Google search returned {len(results)} results for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Google search failed: {e}")
            return []

    @rate_limit(calls=20, period=3600)
    def _search_serpapi(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search using SerpAPI"""
        if not self.config.get('serpapi_key'):
            return []
        
        try:
            endpoint = "https://serpapi.com/search"
            params = {
                "api_key": self.config['serpapi_key'],
                "engine": "google",
                "q": query,
                "num": min(max_results, 20),
                "tbs": "qdr:m6"
            }
            
            response = self.session.get(endpoint, params=params, 
                                      timeout=self.config.get('search_timeout', 30))
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get("organic_results", [])[:max_results]:
                result = {
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "date_published": item.get("date", ""),
                    "source": "serpapi",
                    "relevance_score": 0.85,
                    "domain": extract_domain(item.get("link", "")),
                    "trusted_source": True,
                    "content_richness": 0.85
                }
                results.append(result)
            
            logger.info(f"SerpAPI search returned {len(results)} results for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"SerpAPI search failed: {e}")
            return []

    # Keep all existing helper methods but add missing ones
    def _generate_search_queries(self, query: str, search_type: str) -> List[str]:
        """Generate multiple search queries based on the input type"""
        base_queries = []
        
        if search_type == "company":
            base_queries = [
                f"{query} disruption trends 2024 2025",
                f"{query} competitors threats market analysis",
                f"{query} innovation future technology",
                f"{query} industry challenges opportunities",
                f"companies disrupting {query} business model"
            ]
        elif search_type == "startup":
            base_queries = [
                f"{query} startup funding series market",
                f"{query} competitors technology innovation",
                f"{query} market disruption potential",
                f"startups like {query} similar companies",
                f"{query} business model technology stack"
            ]
        elif search_type == "sector":
            base_queries = [
                f"{query} industry trends 2024 2025 disruption",
                f"{query} market analysis future predictions",
                f"{query} technology innovation breakthrough",
                f"{query} regulatory changes impact",
                f"companies disrupting {query} industry"
            ]
        elif search_type == "geography":
            base_queries = [
                f"{query} economic trends business innovation",
                f"{query} startup ecosystem technology",
                f"{query} market opportunities challenges",
                f"{query} regulatory environment business",
                f"innovation hubs {query} technology companies"
            ]
        elif search_type == "technology":
            base_queries = [
                f"{query} technology trends applications 2024",
                f"{query} market adoption industry impact",
                f"{query} companies implementation case studies",
                f"{query} challenges limitations opportunities",
                f"future of {query} technology predictions"
            ]
        else:
            base_queries = [
                f"{query} market trends analysis 2024",
                f"{query} disruption innovation opportunities",
                f"{query} industry challenges future",
                f"{query} technology impact business",
                f"{query} competitive landscape analysis"
            ]
        
        return base_queries

    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results based on URL and title similarity"""
        seen_urls = set()
        seen_titles = set()
        unique_results = []
        
        for result in results:
            url = result.get('url', '').lower()
            title = result.get('title', '').lower()
            
            # Skip if URL already seen
            if url in seen_urls:
                continue
            
            # Skip if title is very similar to existing one
            title_similar = any(
                self._similarity_ratio(title, seen_title) > 0.8 
                for seen_title in seen_titles
            )
            
            if not title_similar:
                seen_urls.add(url)
                seen_titles.add(title)
                unique_results.append(result)
        
        return unique_results

    def _similarity_ratio(self, str1: str, str2: str) -> float:
        """Calculate similarity ratio between two strings"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, str1, str2).ratio()

    def _enhance_search_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance search results with additional content and metadata"""
        enhanced_results = []
        
        for result in results:
            try:
                # Add domain information if not present
                if 'domain' not in result:
                    result['domain'] = extract_domain(result.get('url', ''))
                
                # Check if domain is in allowed list
                if self.scraping_config.get('allowed_domains'):
                    result['trusted_source'] = result['domain'] in self.scraping_config['allowed_domains']
                elif 'trusted_source' not in result:
                    result['trusted_source'] = True
                
                # Add content richness if not present
                if 'content_richness' not in result:
                    result['content_richness'] = self._calculate_content_richness(result)
                
                # Add timestamp
                result['retrieved_at'] = time.time()
                
                enhanced_results.append(result)
                
            except Exception as e:
                logger.debug(f"Error enhancing result: {e}")
                enhanced_results.append(result)
        
        return enhanced_results

    def _calculate_content_richness(self, result: Dict[str, Any]) -> float:
        """Calculate content richness score for a search result"""
        score = 0.5  # Base score
        
        # Title quality
        title_length = len(result.get("title", ""))
        if 20 <= title_length <= 100:
            score += 0.1
        
        # Snippet quality
        snippet_length = len(result.get("snippet", ""))
        if 50 <= snippet_length <= 300:
            score += 0.2
        
        # Full content availability
        if result.get("full_content"):
            score += 0.3
        
        # Trusted source
        if result.get("trusted_source"):
            score += 0.2
        
        # Recent content
        if result.get("date_published"):
            score += 0.1
        
        return min(1.0, score)

    def __del__(self):
        """Cleanup resources"""
        if self.driver:
            try:
                self.driver.quit()
            except Exception as e:
                logger.debug(f"Error closing WebDriver: {e}")

    def close(self):
        """Explicitly close the WebDriver"""
        if self.driver:
            try:
                self.driver.quit()
                self.driver = None
                logger.info("WebDriver closed successfully")
            except Exception as e:
                logger.error(f"Error closing WebDriver: {e}")
