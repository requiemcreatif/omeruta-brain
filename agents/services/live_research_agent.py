import logging
import time
from typing import Dict, Any, List

from asgiref.sync import sync_to_async

from .enhanced_tinyllama_agent import EnhancedTinyLlamaAgent
from .research_sources import research_sources
from crawler.services import CrawlerService
from crawler.models import CrawlJob
from knowledge_base.services.enhanced_rag import EnhancedRAGService
from knowledge_base.services.pgvector_search import PgVectorSearchService

logger = logging.getLogger(__name__)


class LiveResearchAgent(EnhancedTinyLlamaAgent):
    """Enhanced Research Agent with live internet research capabilities"""

    def __init__(self):
        super().__init__(agent_type="research")
        self.crawler_service = CrawlerService()
        self.rag_service = EnhancedRAGService()
        self.search_service = PgVectorSearchService()

        # Enhanced system prompt for live research
        self.system_prompts[
            "research"
        ] = """You are an advanced research specialist within Omeruta Brain. You excel at:
        - Conducting live internet research to find the most current information
        - Analyzing and synthesizing information from multiple sources (both local and web)
        - Identifying key insights, patterns, and trends
        - Providing comprehensive yet concise research summaries
        - Comparing different perspectives and sources
        - Suggesting related topics and further research directions
        - Citing sources accurately and providing research methodology transparency
        
        When conducting research, you have access to:
        1. Your existing knowledge base (local crawled content)
        2. Live internet research capabilities
        3. Advanced RAG (Retrieval-Augmented Generation) for synthesis
        
        Always be transparent about your research methodology and source reliability."""

    def generate_search_queries(
        self, research_topic: str, max_queries: int = 5
    ) -> List[str]:
        """Generate intelligent search queries for comprehensive research"""
        base_queries = [
            research_topic.strip(),
            f"{research_topic} latest developments",
            f"{research_topic} recent research",
            f"{research_topic} current trends",
            f"{research_topic} 2024 2025",
        ]

        # Add topic-specific query variations
        if any(
            word in research_topic.lower()
            for word in ["technology", "tech", "AI", "software"]
        ):
            base_queries.extend(
                [
                    f"{research_topic} market analysis",
                    f"{research_topic} implementation guide",
                    f"{research_topic} best practices",
                ]
            )
        elif any(
            word in research_topic.lower()
            for word in ["health", "medical", "treatment"]
        ):
            base_queries.extend(
                [
                    f"{research_topic} clinical studies",
                    f"{research_topic} medical research",
                    f"{research_topic} treatment options",
                ]
            )
        elif any(
            word in research_topic.lower()
            for word in ["business", "finance", "economic"]
        ):
            base_queries.extend(
                [
                    f"{research_topic} market trends",
                    f"{research_topic} financial analysis",
                    f"{research_topic} economic impact",
                ]
            )

        return base_queries[:max_queries]

    def generate_research_urls(
        self, search_queries: List[str], max_urls_per_query: int = 3
    ) -> List[str]:
        """Generate URLs for research from multiple sources using flexible configuration"""
        # Use the research sources configuration
        research_urls = research_sources.get_sources_for_topic(
            search_queries, max_sources=15
        )

        logger.info(
            f"Generated {len(research_urls)} research URLs for queries: {search_queries}"
        )
        logger.info(
            f"Primary category detected: {research_sources.get_category_for_topic(search_queries)}"
        )

        return research_urls

    async def conduct_live_research(
        self,
        research_topic: str,
        max_sources: int = 10,
        include_local_kb: bool = True,
        research_depth: str = "comprehensive",
    ) -> Dict[str, Any]:
        """Conduct live internet research on a topic"""

        research_start_time = time.time()
        research_log = []

        try:
            research_log.append(f"🔍 Starting live research on: {research_topic}")

            # Step 1: Search local knowledge base first
            local_results = []
            if include_local_kb:
                research_log.append("📚 Searching local knowledge base...")
                try:
                    # Use sync_to_async to handle database operations
                    search_result = await sync_to_async(
                        self.search_service.enhanced_search
                    )(
                        research_topic,
                        filters={"min_quality": 0.4, "min_relevance": 0.5},
                        use_cache=True,
                    )
                    local_results = search_result.get("results", [])

                except Exception as e:
                    logger.warning(f"Local KB search failed: {e}")
                    local_results = []
                research_log.append(f"   Found {len(local_results)} local sources")

            # Step 2: Generate intelligent search queries
            research_log.append("🧠 Generating research queries...")
            search_queries = self.generate_search_queries(research_topic)
            research_log.append(f"   Generated {len(search_queries)} search queries")

            # Step 3: Generate research URLs
            research_log.append("🌐 Identifying research sources...")
            research_urls = self.generate_research_urls(
                search_queries, max_urls_per_query=2
            )
            research_log.append(f"   Identified {len(research_urls)} potential sources")

            # Step 4: Crawl live content
            research_log.append("🕷️ Crawling live internet sources...")
            await self.crawler_service.start_crawler()

            # Create temporary research job
            temp_research_job = CrawlJob(
                created_by_id=1,  # System user
                start_urls=research_urls[:max_sources],
                strategy="multi",
                max_pages=max_sources,
                use_cache=False,  # Get fresh content
                delay_between_requests=0.3,  # Be respectful
            )

            # Crawl with research-optimized settings
            config_options = {
                "use_content_filter": True,
                "filter_type": "bm25",
                "user_query": research_topic,
                "cache_mode": "bypass",  # Fresh content
            }

            live_pages = await self.crawler_service.crawl_multiple_urls(
                research_urls[:max_sources],
                temp_research_job,
                config_options,
                max_concurrent=3,  # Be respectful to servers
            )

            await self.crawler_service.stop_crawler()

            # Filter successful crawls
            successful_live_pages = [
                p for p in live_pages if p.success and p.clean_markdown
            ]
            research_log.append(
                f"   Successfully crawled {len(successful_live_pages)} sources"
            )

            # Step 5: Combine and analyze all sources
            research_log.append("🔬 Analyzing and synthesizing research...")

            all_sources = []

            # Add local knowledge base results
            for result in local_results:
                content = result.get("chunk_text", "")
                all_sources.append(
                    {
                        "title": result.get("page_title", "Local Knowledge"),
                        "content": content[:500],  # Further reduced to prevent timeout
                        "source_url": result.get("page_url", "Local KB"),
                        "source_type": "local_kb",
                        "relevance_score": 1.0
                        - result.get("distance", 0.5),  # Convert distance to similarity
                        "word_count": len(content.split()),
                        "quality_score": result.get("content_quality_score", 0.5),
                    }
                )

            # Add live research results
            for page in successful_live_pages:
                if page.clean_markdown and len(page.clean_markdown.strip()) > 100:
                    all_sources.append(
                        {
                            "title": page.title or "Web Source",
                            "content": page.clean_markdown[
                                :600
                            ],  # Further reduced to prevent timeout
                            "source_url": page.url,
                            "source_type": "live_web",
                            "relevance_score": 0.8,  # Assume high relevance since it was targeted
                            "word_count": page.word_count or 0,
                            "crawl_time": page.response_time,
                        }
                    )

            research_log.append(f"   Compiled {len(all_sources)} total sources")

            # Step 6: Generate comprehensive research summary
            research_context = self._compile_research_context(
                all_sources, research_topic
            )

            # Filter out empty or irrelevant sources before creating context
            relevant_sources = [
                s
                for s in all_sources
                if s.get("content", "").strip()
                and len(s.get("content", "").strip()) > 50
            ]

            if not relevant_sources:
                research_response = f"I was unable to find relevant information about '{research_topic}' in the available sources. The sources found were not relevant to your question."
            else:
                # Create focused context from relevant sources only
                focused_context = self._compile_research_context(
                    relevant_sources, research_topic
                )

                research_prompt = f"""You are answering this specific question: "{research_topic}"

Based on the research sources below, provide a direct and factual answer. Do not describe what the sources contain - instead, extract the actual information to answer the question.

Research Sources:
{focused_context[:2500]}

Instructions:
1. Answer the question directly using information from the sources
2. If sources contain the answer, state it clearly
3. If sources don't contain the answer, say so clearly
4. Do not describe source contents - use the information within them
5. Be factual and specific

Answer the question now:"""

                try:
                    research_response = self.llm_service.generate_response(
                        prompt=research_prompt,
                        max_tokens=500,
                        system_prompt="You are a factual research assistant. Always provide direct, clear answers to questions based on available sources. Do not hedge or avoid answering when sources contain relevant information.",
                    )
                except Exception as e:
                    logger.error(f"Error generating response: {e}")
                    research_response = None

            research_time = time.time() - research_start_time
            research_log.append(f"✅ Research completed in {research_time:.2f} seconds")

            final_response = (
                research_response
                or f"I encountered an issue generating a response from the research data. Please try rephrasing your question or try again."
            )

            return {
                "status": "success",
                "research_topic": research_topic,
                "response": final_response,
                "sources": all_sources,
                "research_methodology": {
                    "search_queries_used": search_queries,
                    "urls_attempted": len(research_urls),
                    "sources_crawled": len(successful_live_pages),
                    "local_sources": len(local_results),
                    "total_sources": len(all_sources),
                    "research_time_seconds": research_time,
                    "research_depth": research_depth,
                },
                "research_log": research_log,
                "quality_metrics": {
                    "source_diversity": len(set(s["source_type"] for s in all_sources)),
                    "content_freshness": "live" if successful_live_pages else "cached",
                    "total_content_analyzed": sum(s["word_count"] for s in all_sources),
                },
            }

        except Exception as e:
            logger.error(f"Live research failed: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "research_topic": research_topic,
                "research_log": research_log,
                "response": "I encountered an error while conducting live research. Let me try using the local knowledge base instead.",
            }

    def _compile_research_context(self, sources: List[Dict], topic: str) -> str:
        """Compile research context from all sources"""
        context_parts = []

        for i, source in enumerate(sources, 1):
            source_type = (
                "🌐 Web" if source["source_type"] == "live_web" else "📚 Local"
            )
            context_parts.append(
                f"""
Source {i} ({source_type}):
Title: {source['title']}
URL: {source['source_url']}
Relevance: {source['relevance_score']:.2f}
Content: {source['content'][:400]}...
"""
            )

        return "\n".join(context_parts)

    async def enhanced_research_chat(
        self,
        message: str,
        use_live_research: bool = True,
        max_sources: int = 8,
        research_depth: str = "comprehensive",
    ) -> Dict[str, Any]:
        """Enhanced chat with live research capabilities"""

        # Determine if this requires research
        research_indicators = [
            "research",
            "find information",
            "latest",
            "current",
            "recent",
            "what's new",
            "developments",
            "trends",
            "analysis",
            "investigate",
            "explore",
            "study",
            "examine",
            "survey",
            "review",
        ]

        needs_research = any(
            indicator in message.lower() for indicator in research_indicators
        )

        if needs_research and use_live_research:
            # Conduct live research
            research_result = await self.conduct_live_research(
                research_topic=message,
                max_sources=max_sources,
                research_depth=research_depth,
            )

            if research_result["status"] == "success":
                return {
                    "response": research_result["response"],
                    "model_used": "tinyllama",
                    "research_conducted": True,
                    "live_sources_used": len(
                        [
                            s
                            for s in research_result["sources"]
                            if s["source_type"] == "live_web"
                        ]
                    ),
                    "local_sources_used": len(
                        [
                            s
                            for s in research_result["sources"]
                            if s["source_type"] == "local_kb"
                        ]
                    ),
                    "sources": research_result["sources"],
                    "research_methodology": research_result["research_methodology"],
                    "quality_metrics": research_result["quality_metrics"],
                    "research_log": research_result["research_log"],
                    "agent_type": "live_research",
                }
            else:
                # Fallback to regular processing
                return self.process_message(
                    message=message,
                    use_context=True,
                    response_config={"max_tokens": 500},
                )
        else:
            # Use regular research agent processing
            return self.process_message(
                message=message, use_context=True, response_config={"max_tokens": 500}
            )

    def get_research_capabilities(self) -> Dict[str, Any]:
        """Get information about research capabilities"""
        return {
            "live_research": True,
            "supported_sources": [
                "Google Scholar",
                "ArXiv",
                "Wikipedia",
                "Reuters",
                "BBC",
                "Nature",
                "ScienceDirect",
                "PubMed",
                "GitHub",
                "Stack Overflow",
            ],
            "research_strategies": ["comprehensive", "focused", "rapid"],
            "max_sources_per_query": 20,
            "supports_local_kb": True,
            "supports_live_web": True,
            "research_depth_options": ["surface", "comprehensive", "deep"],
            "content_types": ["academic", "news", "technical", "general"],
        }
