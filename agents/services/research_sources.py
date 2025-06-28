"""
Research Sources Configuration
Manages URL sources for different research topics and categories.
"""

from typing import Dict, List, Set
import re


class ResearchSourcesConfig:
    """Configuration for research sources organized by topic categories"""

    def __init__(self):
        self.topic_patterns = {
            # Political/Government topics
            "politics": {
                "keywords": [
                    "president",
                    "election",
                    "government",
                    "politics",
                    "congress",
                    "senate",
                    "white house",
                    "biden",
                    "trump",
                    "democrat",
                    "republican",
                    "vote",
                    "campaign",
                    "policy",
                    "administration",
                ],
                "sources": [
                    "https://www.reuters.com/world/us/",
                    "https://www.bbc.com/news/world-us-canada",
                    "https://apnews.com/hub/politics",
                    "https://www.politico.com/",
                    "https://en.wikipedia.org/wiki/President_of_the_United_States",
                    "https://en.wikipedia.org/wiki/2024_United_States_presidential_election",
                    "https://www.cnn.com/politics",
                    "https://www.npr.org/sections/politics/",
                ],
            },
            # Technology/AI topics
            "technology": {
                "keywords": [
                    "ai",
                    "artificial intelligence",
                    "machine learning",
                    "deep learning",
                    "technology",
                    "tech",
                    "software",
                    "programming",
                    "development",
                    "computer",
                    "digital",
                    "innovation",
                    "startup",
                ],
                "sources": [
                    "https://en.wikipedia.org/wiki/Artificial_intelligence",
                    "https://en.wikipedia.org/wiki/Machine_learning",
                    "https://techcrunch.com/",
                    "https://www.theverge.com/tech",
                    "https://arstechnica.com/",
                    "https://www.wired.com/category/business/tech/",
                    "https://www.reuters.com/technology/",
                    "https://www.bbc.com/news/technology",
                    "https://stackoverflow.blog/",
                    "https://github.blog/",
                ],
            },
            # Health/Medical topics
            "health": {
                "keywords": [
                    "health",
                    "medical",
                    "medicine",
                    "disease",
                    "treatment",
                    "doctor",
                    "hospital",
                    "clinical",
                    "research",
                    "drug",
                    "vaccine",
                    "therapy",
                    "patient",
                    "healthcare",
                    "wellness",
                ],
                "sources": [
                    "https://www.who.int/news",
                    "https://www.cdc.gov/media/releases/",
                    "https://www.nature.com/subjects/medical-research",
                    "https://www.reuters.com/business/healthcare-pharmaceuticals/",
                    "https://www.bbc.com/news/health",
                    "https://www.webmd.com/news/",
                    "https://en.wikipedia.org/wiki/Medicine",
                    "https://pubmed.ncbi.nlm.nih.gov/",
                ],
            },
            # Business/Finance topics
            "business": {
                "keywords": [
                    "business",
                    "finance",
                    "economy",
                    "market",
                    "stock",
                    "investment",
                    "company",
                    "corporate",
                    "earnings",
                    "profit",
                    "revenue",
                    "trade",
                    "economic",
                    "financial",
                    "banking",
                    "cryptocurrency",
                ],
                "sources": [
                    "https://www.reuters.com/business/",
                    "https://www.bbc.com/news/business",
                    "https://www.bloomberg.com/",
                    "https://finance.yahoo.com/news/",
                    "https://www.cnbc.com/business/",
                    "https://www.wsj.com/",
                    "https://www.ft.com/",
                    "https://en.wikipedia.org/wiki/Economics",
                ],
            },
            # Science topics
            "science": {
                "keywords": [
                    "science",
                    "research",
                    "study",
                    "discovery",
                    "experiment",
                    "scientist",
                    "physics",
                    "chemistry",
                    "biology",
                    "climate",
                    "environment",
                    "space",
                    "astronomy",
                    "quantum",
                    "genetics",
                    "evolution",
                ],
                "sources": [
                    "https://www.nature.com/news",
                    "https://www.sciencedaily.com/",
                    "https://www.science.org/news",
                    "https://www.newscientist.com/",
                    "https://www.reuters.com/business/environment/",
                    "https://www.bbc.com/news/science-environment",
                    "https://en.wikipedia.org/wiki/Science",
                    "https://www.nasa.gov/news/",
                ],
            },
            # Sports topics
            "sports": {
                "keywords": [
                    "sports",
                    "football",
                    "basketball",
                    "baseball",
                    "soccer",
                    "tennis",
                    "olympics",
                    "athlete",
                    "team",
                    "game",
                    "match",
                    "championship",
                    "league",
                    "player",
                    "coach",
                ],
                "sources": [
                    "https://www.espn.com/",
                    "https://www.reuters.com/lifestyle/sports/",
                    "https://www.bbc.com/sport",
                    "https://www.cnn.com/sport",
                    "https://en.wikipedia.org/wiki/Sport",
                ],
            },
            # Entertainment topics
            "entertainment": {
                "keywords": [
                    "entertainment",
                    "movie",
                    "film",
                    "music",
                    "celebrity",
                    "actor",
                    "actress",
                    "director",
                    "hollywood",
                    "television",
                    "tv",
                    "show",
                    "concert",
                    "album",
                    "streaming",
                ],
                "sources": [
                    "https://www.reuters.com/lifestyle/entertainment/",
                    "https://www.bbc.com/news/entertainment-arts",
                    "https://variety.com/",
                    "https://www.hollywoodreporter.com/",
                    "https://en.wikipedia.org/wiki/Entertainment",
                ],
            },
        }

        # General news sources for broad topics
        self.general_news_sources = [
            "https://www.reuters.com/",
            "https://www.bbc.com/news",
            "https://apnews.com/",
            "https://www.cnn.com/",
            "https://www.npr.org/",
            "https://en.wikipedia.org/wiki/Main_Page",
        ]

        # Time-sensitive sources for recent/current topics
        self.current_news_sources = [
            "https://www.reuters.com/world/",
            "https://www.bbc.com/news/",
            "https://apnews.com/",
            "https://www.cnn.com/",
            "https://www.npr.org/sections/news/",
        ]

    def get_sources_for_topic(
        self, search_queries: List[str], max_sources: int = 15
    ) -> List[str]:
        """
        Get relevant sources based on search queries

        Args:
            search_queries: List of search query strings
            max_sources: Maximum number of sources to return

        Returns:
            List of relevant URLs for the topic
        """
        combined_query = " ".join(search_queries).lower()
        relevant_sources = set()

        # Check for time-sensitive queries first
        if any(
            word in combined_query
            for word in ["2024", "2025", "recent", "latest", "current", "today", "now"]
        ):
            relevant_sources.update(self.current_news_sources)

        # Find matching topic categories
        matched_categories = []
        for category, config in self.topic_patterns.items():
            keyword_matches = sum(
                1 for keyword in config["keywords"] if keyword in combined_query
            )
            if keyword_matches > 0:
                matched_categories.append((category, keyword_matches))

        # Sort by relevance (number of keyword matches)
        matched_categories.sort(key=lambda x: x[1], reverse=True)

        # Add sources from matched categories
        for category, _ in matched_categories[:3]:  # Top 3 most relevant categories
            relevant_sources.update(self.topic_patterns[category]["sources"])

        # If no specific matches, add general news sources
        if not matched_categories:
            relevant_sources.update(self.general_news_sources)

        # Convert to list and limit
        sources_list = list(relevant_sources)
        return sources_list[:max_sources]

    def get_category_for_topic(self, search_queries: List[str]) -> str:
        """
        Determine the primary category for a set of search queries

        Returns:
            The most relevant category name or 'general'
        """
        combined_query = " ".join(search_queries).lower()

        best_category = "general"
        best_score = 0

        for category, config in self.topic_patterns.items():
            score = sum(
                1 for keyword in config["keywords"] if keyword in combined_query
            )
            if score > best_score:
                best_score = score
                best_category = category

        return best_category

    def add_custom_sources(self, category: str, sources: List[str]):
        """Add custom sources to a category"""
        if category not in self.topic_patterns:
            self.topic_patterns[category] = {"keywords": [], "sources": []}

        self.topic_patterns[category]["sources"].extend(sources)

    def get_all_categories(self) -> List[str]:
        """Get list of all available categories"""
        return list(self.topic_patterns.keys())


# Global instance
research_sources = ResearchSourcesConfig()
