# Omeruta Brain Crawler Implementation

## Overview

This crawler is built using Crawl4AI and is designed to extract clean markdown content optimized for vector databases and LLM processing. It provides comprehensive web crawling capabilities with various strategies and clean content extraction.

## Features

### Core Features

- **Multiple Crawl Strategies**: Single URL, Deep BFS, Deep DFS, and Best-First crawling
- **Clean Markdown Extraction**: Optimized content for vector databases
- **Content Filtering**: Removes navigation, ads, and irrelevant content
- **Link Extraction**: Tracks internal and external links
- **Metadata Extraction**: Title, description, keywords, author, publish date
- **Progress Tracking**: Real-time crawl progress and statistics
- **Error Handling**: Comprehensive error tracking and retry mechanisms
- **Caching**: Built-in caching for efficient re-crawling

### API Endpoints

#### Crawl Job Management

- `POST /api/crawler/jobs/create/` - Create new crawl job
- `GET /api/crawler/jobs/` - List all crawl jobs
- `GET /api/crawler/jobs/{id}/` - Get specific crawl job details
- `POST /api/crawler/jobs/{id}/stop/` - Stop running crawl job
- `GET /api/crawler/jobs/{id}/statistics/` - Get crawl statistics

#### Content Access

- `GET /api/crawler/jobs/{job_id}/pages/` - List crawled pages for a job
- `GET /api/crawler/pages/{id}/` - Get specific page details
- `POST /api/crawler/extract/` - Extract content from single URL (no persistent job)
- `POST /api/crawler/extract/bulk/` - Extract content from multiple URLs

#### Health & Monitoring

- `GET /api/crawler/health/` - Check crawler health
- `GET /api/crawler/stats/` - Get system-wide crawler statistics

## Models

### CrawlJob

Tracks crawl jobs with configuration, progress, and results:

- Strategy (single, deep_bfs, deep_dfs, deep_best)
- Start URLs and crawl limits
- Include/exclude patterns
- Progress tracking and statistics

### CrawledPage

Stores individual crawled pages:

- Raw and clean markdown content
- Metadata (title, description, keywords, etc.)
- Links and content statistics
- Vector processing flags

### CrawlStatistics

Aggregated statistics for crawl jobs:

- Content and link statistics
- Performance metrics
- Quality scores

## Usage Examples

### Creating a Crawl Job

```python
import requests

# Create a single URL crawl
response = requests.post('http://localhost:8000/api/crawler/jobs/create/', {
    'strategy': 'single',
    'start_urls': ['https://example.com'],
    'max_pages': 1,
    'delay_between_requests': 0.5,
    'respect_robots_txt': True,
    'use_cache': True
}, headers={'Authorization': 'Bearer YOUR_JWT_TOKEN'})
```

### Deep Crawling

```python
# Create a deep crawl job
response = requests.post('http://localhost:8000/api/crawler/jobs/create/', {
    'strategy': 'deep_bfs',
    'start_urls': ['https://example.com'],
    'max_pages': 50,
    'max_depth': 3,
    'include_patterns': ['example.com/*'],
    'exclude_patterns': ['*/admin/*', '*/login/*'],
    'delay_between_requests': 1.0,
    'respect_robots_txt': True,
    'use_cache': True
}, headers={'Authorization': 'Bearer YOUR_JWT_TOKEN'})
```

### Quick Content Extraction

```python
# Extract content without creating a persistent job
response = requests.post('http://localhost:8000/api/crawler/extract/', {
    'url': 'https://example.com/article',
    'extract_links': True,
    'fit_markdown': True
}, headers={'Authorization': 'Bearer YOUR_JWT_TOKEN'})
```

## Management Commands

### Test Crawler

```bash
python manage.py test_crawler https://example.com --strategy=single --user=admin
```

### Deep Crawl Test

```bash
python manage.py test_crawler https://example.com --strategy=deep_bfs --max-pages=10 --max-depth=2
```

## Configuration

### Django Settings

The crawler is automatically configured when you add `'crawler'` to `INSTALLED_APPS`.

### Crawl4AI Configuration

The crawler uses optimized Crawl4AI settings:

- Browser: Chromium with stealth mode
- Content filtering: BM25 and pruning filters
- Markdown generation: Optimized for LLMs
- Caching: SQLite-based caching

## Content Quality

### Clean Markdown Features

- Removes navigation, ads, and boilerplate content
- Preserves article structure and formatting
- Extracts semantic content only
- Optimized for vector embeddings
- Maintains readability for LLMs

### Content Filtering

- BM25 relevance scoring
- Pruning of low-value content
- Duplicate content detection
- Language detection and filtering

## Vector Database Integration

The crawler is designed to integrate seamlessly with vector databases:

1. **Clean Content**: All content is cleaned and optimized for embeddings
2. **Chunk Preparation**: Content is prepared for chunking strategies
3. **Metadata Preservation**: Rich metadata for context
4. **Processing Flags**: Tracks vector processing status

## Admin Interface

Comprehensive Django admin interface provides:

- Visual progress tracking
- Content preview and editing
- Link management
- Statistics and performance metrics
- Bulk operations and filtering

## Security Features

- User-based access control
- Rate limiting and respectful crawling
- Robots.txt compliance
- Content sanitization
- Error logging and monitoring

## Performance

- Asynchronous crawling with Crawl4AI
- Efficient caching mechanisms
- Database optimization with indexes
- Pagination for large result sets
- Background job processing ready

## Future Enhancements

- Celery integration for background processing
- Real-time WebSocket updates
- Advanced content analysis
- Machine learning-based content scoring
- Distributed crawling capabilities
