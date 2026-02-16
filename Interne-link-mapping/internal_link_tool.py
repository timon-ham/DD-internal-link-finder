
import subprocess
import sys
import logging
import time
import re
import warnings

# --- 1. Silent Dependency Install ---
def install_deps():
    packages = ["requests", "beautifulsoup4", "pandas", "scikit-learn", "openpyxl", "openai", "numpy"]
    try:
        import requests, bs4, pandas, openai, numpy, sklearn, openpyxl
    except ImportError:
        print("Installing dependencies...", end=" ")
        for p in packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", p], stdout=subprocess.DEVNULL)
            except: pass
        print("Done.")

install_deps()

# --- 2. Imports & Setup ---
import requests
import pandas as pd
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import xml.etree.ElementTree as ET

logging.getLogger().setLevel(logging.CRITICAL)
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
# Replace these with your actual values or keep them as is if you modify them later
ROOT_URL = "https://example.com" 
TARGET_URL_TO_BOOST = "https://example.com/target-page"
ANCHOR_TEXTS_INPUT = "keyword one, keyword two, keyword three, keyword four, keyword five"
OPENAI_API_KEY = "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"

POSSIBLE_ANCHORS = [x.strip() for x in ANCHOR_TEXTS_INPUT.split(',') if x.strip()]

print(f"Targeting: {len(POSSIBLE_ANCHORS)} anchor texts")
print(f"Target Page: {TARGET_URL_TO_BOOST}")

# --- 3. Logic Classes ---
class Crawler:
    def __init__(self, delay: float = 0.05, max_pages: int = 300):
        self.delay = delay
        self.visited = set()
        self.results = []
        self.max_pages = max_pages

    def _is_valid_url(self, url: str, base_domain: str) -> bool:
        parsed = urlparse(url)
        # 1. Check Domain & Scheme
        if parsed.netloc != base_domain or parsed.scheme not in ['http', 'https']:
            return False
        
        # 2. Check Exclusions (? and /page/)
        if '?' in url or '/page/' in url:
            return False
            
        # 3. Check Extensions
        if any(url.lower().endswith(e) for e in ['.jpg', '.png', '.pdf', '.css', '.js', '.gif', '.svg', '.xml', '.zip']):
            return False
            
        return True

    def _fetch_sitemap_urls(self, base_url: str) -> set:
        sitemaps = set()
        candidates = [
            urljoin(base_url, '/sitemap.xml'),
            urljoin(base_url, '/sitemap_index.xml'),
            urljoin(base_url, '/wp-sitemap.xml')
        ]
        for sm_url in candidates:
            try:
                resp = requests.get(sm_url, timeout=5)
                if resp.status_code == 200:
                    try:
                        root = ET.fromstring(resp.content)
                        for child in root.iter():
                             if 'loc' in child.tag and child.text:
                                sitemaps.add(child.text.strip())
                    except: pass
            except: pass
        return sitemaps

    def crawl(self, start_url: str) -> pd.DataFrame:
        self.visited = set()
        self.results = []
        base_domain = urlparse(start_url).netloc
        queue = [start_url]
        self.visited.add(start_url)
        
        print("Looking for sitemaps...", end=" ")
        sitemaps = self._fetch_sitemap_urls(start_url)
        print(f"Found {len(sitemaps)} URLs.")
        
        for u in sitemaps:
            if self._is_valid_url(u, base_domain) and u not in self.visited:
                self.visited.add(u)
                queue.append(u)
        
        print(f"Crawling up to {self.max_pages} pages...")
        count = 0
        last_print = 0
        
        while queue and len(self.results) < self.max_pages:
            url = queue.pop(0)
            count += 1
            if count - last_print >= 20:
                print(f"Processed {count} URLs...", end="\r")
                last_print = count

            try:
                time.sleep(self.delay)
                resp = requests.get(url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
                if resp.status_code == 200 and 'text/html' in resp.headers.get('Content-Type', ''):
                    # Check noindex
                    if 'noindex' in resp.headers.get('X-Robots-Tag', '').lower(): continue
                    
                    soup = BeautifulSoup(resp.text, 'html.parser')
                    meta = soup.find('meta', attrs={'name': 'robots'})
                    if meta and 'noindex' in meta.get('content', '').lower(): continue

                    self.results.append({'url': url, 'html': resp.text})
                    
                    for link in soup.find_all('a', href=True):
                        next_url = urljoin(url, link['href']).split('#')[0]
                        if self._is_valid_url(next_url, base_domain) and next_url not in self.visited:
                            self.visited.add(next_url)
                            queue.append(next_url)
            except: pass
            
        print(f"\nCrawl complete. Scanned {len(self.results)} pages.")
        return pd.DataFrame(self.results)

def clean_soup(soup):
    """Removes boilerplate elements from soup in-place."""
    # STRICT REMOVAL of boilerplate
    # Included 'header' and 'footer'
    for x in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'noscript', 'meta', 'link']):
        x.decompose()
    return soup

def get_embedding(text, client):
    return client.embeddings.create(input=[text.replace("\n", " ")], model="text-embedding-3-small").data[0].embedding

# --- 4. Main Execution ---

if 'ROOT_URL' not in globals() or not ROOT_URL:
    print("❌ Error: Please set the ROOT_URL variable!")
else:
    client = OpenAI(api_key=OPENAI_API_KEY)
    crawler = Crawler()
    df = crawler.crawl(ROOT_URL)
    
    if not df.empty:
        print("Processing content... (this may take a minute)")
        
        # Get Target Info
        target_text = ""
        try:
            t_row = df[df['url'] == TARGET_URL_TO_BOOST]
            if not t_row.empty:
                t_soup = BeautifulSoup(t_row.iloc[0]['html'], 'html.parser')
                clean_soup(t_soup)
                target_text = t_soup.get_text(separator=' ', strip=True)
            else:
                resp = requests.get(TARGET_URL_TO_BOOST, headers={'User-Agent': 'Mozilla/5.0'})
                t_soup = BeautifulSoup(resp.text, 'html.parser')
                clean_soup(t_soup)
                target_text = t_soup.get_text(separator=' ', strip=True)
        except Exception as e:
            print(f"⚠️ Error fetching target page: {e}")
            
        if target_text:
            target_vec = get_embedding(target_text[:15000], client)
            
            # Results Containers
            opportunities = []
            reviews = []
            
            target_path = urlparse(TARGET_URL_TO_BOOST).path
            normalized_anchors = [p.lower() for p in POSSIBLE_ANCHORS]

            print("Analyzing pages...")
            for i, row in df.iterrows():
                current_url = row['url']
                if current_url == TARGET_URL_TO_BOOST: continue
                
                html = row['html']
                soup = BeautifulSoup(html, 'html.parser')
                
                # --- CRITICAL FIX: Clean the soup BEFORE checking for links ---
                clean_soup(soup)
                # ----------------------------------------------------------------
                
                # --- Check for Existing Links (in main content only) ---
                has_link = False
                link_anchor = ""
                is_exact_match_link = False
                
                # Find links to target
                for a in soup.find_all('a', href=True):
                    href = a['href']
                    abs_href = urljoin(current_url, href).split('#')[0]
                    
                    if abs_href == TARGET_URL_TO_BOOST or href == target_path:
                        has_link = True
                        anchor = a.get_text(strip=True).lower()
                        if anchor in normalized_anchors:
                            is_exact_match_link = True
                            break # Keyword link found, ignore page
                        else:
                            link_anchor = anchor
                
                # --- Decision Logic ---
                if is_exact_match_link:
                    # Case: Link exists AND matches keyword.
                    # Action: Ignore page.
                    continue
                
                elif has_link:
                    # Case: Link exists BUT didn't match keywords.
                    # Action: Review.
                    reviews.append({
                        'Page URL': current_url,
                        'Existing Link Anchor': link_anchor,
                        'Action': 'Review: Change anchor?'
                    })
                    
                else:
                    # Case: No link exists in MAIN content.
                    # Action: Standard Analysis
                    
                    # --- NEW: Ignore headings and links for text analysis ---
                    # We decompose these tags so their text is NOT included in the search
                    for tag in soup(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'a']):
                        tag.decompose()
                    # -----------------------------------------------------

                    text = soup.get_text(separator=' ', strip=True)[:15000]
                    if not text: continue

                    # A. Keyword Match
                    matches = [k for k in POSSIBLE_ANCHORS if k.lower() in text.lower()]
                    
                    # B. Semantic Match
                    score = 0
                    try:
                        vec = get_embedding(text, client)
                        score = cosine_similarity([target_vec], [vec])[0][0]
                    except: pass
                    
                    if matches or score > 0.73: # Slightly lowered threshold
                        opportunities.append({
                            'Page URL': current_url,
                            'Target URL': TARGET_URL_TO_BOOST,
                            'Keywords Found': ", ".join(matches),
                            'Semantic Score': round(score, 4),
                            'Type': 'Keyword' if matches else 'Semantic'
                        })

            # --- Export ---
            if opportunities or reviews:
                filename = "link_opportunities_v3.xlsx"
                try:
                    with pd.ExcelWriter(filename) as writer:
                        if opportunities:
                            pd.DataFrame(opportunities).sort_values('Semantic Score', ascending=False).to_excel(writer, sheet_name='New Opportunities', index=False)
                        if reviews:
                            pd.DataFrame(reviews).to_excel(writer, sheet_name='Existing Link Review', index=False)
                    
                    print(f"\n✅ Done! Found {len(opportunities)} new opportunities and {len(reviews)} pages to review.")
                    print(f"Results saved to {filename}")
                except Exception as e:
                    print(f"Error saving file: {e}")
            else:
                print("\nℹ️ No matching opportunities or reviews found.")

        else:
            print("❌ Could not analyze target page content.")
    else:
        print("❌ No pages found to analyze.")
