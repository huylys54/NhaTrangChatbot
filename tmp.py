import asyncio
from urllib.parse import urldefrag
from crawl4ai import (
    AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode,
    MemoryAdaptiveDispatcher
)
from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter

from typing import List
from config import CRAWLER_URLS
import hashlib
import os

async def crawl_internal_urls(init_url: str) -> List[str]:
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url=init_url,
        )
        if result.success:
            save_to_markdown(result, './data/tmp')
        return [u['href'] for u in result.links['internal']]
    
def save_to_markdown(result: List[str], directory="./data/raw/nhatrang.khanhhoa.gov.vn"):

    os.makedirs(directory, exist_ok=True)
    
    url_hash = hashlib.md5(result.url.encode('utf-8')).hexdigest()  # Use md5 or sha256
    filename = f"{url_hash}.md"
    path = os.path.join(directory, filename)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"# {result.url}\n\n")
        f.write(f"## Depth: {result.metadata.get('depth', 0)}\n\n")
        f.write(result.markdown) 


async def crawl_recursive_batch(start_urls, max_depth=2, max_concurrent=10):
    browser_config = BrowserConfig(headless=True, verbose=False)
    
    url_filter = URLPatternFilter(patterns=["!*twitter.com*", "!*login*", "!*facebook.com*", "!*youtube.com*"])

    
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        exclude_domains=["youtube.com", "facebook.com", "twitter.com"],
        stream=False
    )
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,      # Don't exceed 70% memory usage
        check_interval=1.0,                 # Check memory every second
        max_session_permit=max_concurrent   # Max parallel browser sessions
    )

    # Track visited URLs to prevent revisiting and infinite loops (ignoring fragments)
    visited = set()
    def normalize_url(url):
        # Remove fragment (part after #)
        return urldefrag(url)[0]
    current_urls = set([normalize_url(u) for u in start_urls])
    
   
    async with AsyncWebCrawler(config=browser_config) as crawler:
        for depth in range(max_depth):
            print(f"\n=== Crawling Depth {depth+1} ===")
            # Only crawl URLs we haven't seen yet (ignoring fragments)
            urls_to_crawl = [normalize_url(url) for url in current_urls if normalize_url(url) not in visited]

            if not urls_to_crawl:
                break

            # Batch-crawl all URLs at this depth in parallel
            results = await crawler.arun_many(
                urls=urls_to_crawl,
                config=run_config,
                
                dispatcher=dispatcher
            )

            next_level_urls = set()

            for result in results:
                norm_url = normalize_url(result.url)
                visited.add(norm_url)  # Mark as visited (no fragment)
                if result.success:
                    save_to_markdown(result)
                    print(f"[OK] {result.url} | Markdown: {len(result.markdown) if result.markdown else 0} chars")
                    # Collect all new internal links for the next depth
                    for link in result.links.get("external", []):
                        next_url = normalize_url(link["href"])
                        if next_url not in visited and next_url != "https://nhatrang.khanhhoa.gov.vn":
                            next_level_urls.add(next_url)
                else:
                    print(f"[ERROR] {result.url}: {result.error_message}")
                    
            # Move to the next set of URLs for the next recursion depth
            current_urls = next_level_urls
            

if __name__ == "__main__":
    asyncio.run(crawl_internal_urls("https://nhatrang.khanhhoa.gov.vn/tour-du-lich"))
    # asyncio.run(crawl_recursive_batch(CRAWLER_URLS, max_depth=3, max_concurrent=10))