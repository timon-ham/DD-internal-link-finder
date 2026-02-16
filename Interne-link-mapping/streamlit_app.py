import streamlit as st
import requests
import pandas as pd
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import xml.etree.ElementTree as ET
import time
import io

# --- Configuration & Setup ---
st.set_page_config(page_title="Internal Link Suggester", page_icon="🔗", layout="wide")

st.title("🔗 Internal Link Opportunity Finder")
st.markdown("""
This tool crawls your website and suggests internal linking opportunities based on **keyword matching** and **semantic similarity**.
""")

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")
    
    # Hardcoded API Key as requested
    openai_api_key = "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"
    
    st.divider()
    
    max_pages = st.number_input("Max Pages to Crawl", min_value=10, max_value=1000, value=100, step=50)
    crawl_delay = st.number_input("Crawl Delay (seconds)", min_value=0.0, max_value=5.0, value=0.05, step=0.05)
    
    st.divider()
    
    st.markdown("### How it works")
    st.markdown("""
    1. **Crawls** the website starting from the Root URL.
    2. **Analyzes** the content of each page.
    3. **Compares** content with your Target Page.
    4. **Suggests** links if keywords are found or content is semantically similar.
    """)

# --- Main Inputs ---
col1, col2 = st.columns(2)

with col1:
    root_url = st.text_input("Root URL", placeholder="https://example.com")

with col2:
    target_url_to_boost = st.text_input("Target Page URL (to boost)", placeholder="https://example.com/important-page")

anchor_texts_input = st.text_area("Target Anchor Texts (comma separated)", placeholder="keyword 1, keyword 2, keyword 3", help="Pages containing these keywords will be prioritized.")

start_analysis = st.button("Start Analysis", type="primary")

# --- Classes & Functions ---

class Crawler:
    def __init__(self, delay: float = 0.05, max_pages: int = 300):
        self.delay = delay
        self.visited = set()
        self.results = []
        self.max_pages = max_pages
        self.progress_bar = None
        self.status_text = None

    def _is_valid_url(self, url: str, base_domain: str) -> bool:
        parsed = urlparse(url)
        if parsed.netloc != base_domain or parsed.scheme not in ['http', 'https']:
            return False
        if '?' in url or '/page/' in url:
            return False
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

    def crawl(self, start_url: str):
        self.visited = set()
        self.results = []
        base_domain = urlparse(start_url).netloc
        queue = [start_url]
        self.visited.add(start_url)
        
        status_container = st.empty()
        status_container.info("Looking for sitemaps...")
        
        sitemaps = self._fetch_sitemap_urls(start_url)
        status_container.success(f"Found {len(sitemaps)} URLs from sitemaps.")
        time.sleep(1)
        
        for u in sitemaps:
            if self._is_valid_url(u, base_domain) and u not in self.visited:
                self.visited.add(u)
                queue.append(u)
        
        status_container.info(f"Crawling up to {self.max_pages} pages...")
        progress_bar = st.progress(0)
        
        count = 0
        
        while queue and len(self.results) < self.max_pages:
            url = queue.pop(0)
            count += 1
            
            # Update progress
            progress = min(len(self.results) / self.max_pages, 1.0)
            progress_bar.progress(progress)
            status_container.text(f"Scanning: {url}")

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
            
        progress_bar.empty()
        status_container.empty()
        return pd.DataFrame(self.results)

def clean_soup(soup):
    """Removes boilerplate elements from soup in-place."""
    for x in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'noscript', 'meta', 'link']):
        x.decompose()
    return soup

def get_embedding(text, client):
    try:
        return client.embeddings.create(input=[text.replace("\n", " ")], model="text-embedding-3-small").data[0].embedding
    except Exception as e:
        st.error(f"Error generating embedding: {e}")
        return None

# --- Main Logic ---

if start_analysis:
    if not openai_api_key:
        st.error("Please provide an OpenAI API Key in the sidebar.")
    elif not root_url or not target_url_to_boost:
        st.error("Please provide both Root URL and Target URL.")
    else:
        client = OpenAI(api_key=openai_api_key)
        crawler = Crawler(delay=crawl_delay, max_pages=max_pages)
        
        with st.spinner("Crawling website..."):
            df = crawler.crawl(root_url)
            
        if not df.empty:
            st.success(f"Crawl complete. Scanned {len(df)} pages.")
            
            with st.spinner("Analyzing content & Calculating Similarity..."):
                # Get Target Info
                target_text = ""
                try:
                    t_row = df[df['url'] == target_url_to_boost]
                    if not t_row.empty:
                        t_soup = BeautifulSoup(t_row.iloc[0]['html'], 'html.parser')
                        clean_soup(t_soup)
                        target_text = t_soup.get_text(separator=' ', strip=True)
                    else:
                        resp = requests.get(target_url_to_boost, headers={'User-Agent': 'Mozilla/5.0'})
                        t_soup = BeautifulSoup(resp.text, 'html.parser')
                        clean_soup(t_soup)
                        target_text = t_soup.get_text(separator=' ', strip=True)
                except Exception as e:
                    st.error(f"Error fetching target page: {e}")
                    target_text = None

                if target_text:
                    target_vec = get_embedding(target_text[:15000], client)
                    
                    if target_vec:
                        possible_anchors = [x.strip() for x in anchor_texts_input.split(',') if x.strip()]
                        normalized_anchors = [p.lower() for p in possible_anchors]
                        target_path = urlparse(target_url_to_boost).path
                        
                        opportunities = []
                        reviews = []
                        
                        progress_bar = st.progress(0)
                        
                        for i, row in df.iterrows():
                            progress_bar.progress((i + 1) / len(df))
                            
                            current_url = row['url']
                            if current_url == target_url_to_boost: continue
                            
                            html = row['html']
                            soup = BeautifulSoup(html, 'html.parser')
                            clean_soup(soup)
                            
                            # Check for Existing Links
                            has_link = False
                            link_anchor = ""
                            is_exact_match_link = False
                            
                            for a in soup.find_all('a', href=True):
                                href = a['href']
                                abs_href = urljoin(current_url, href).split('#')[0]
                                
                                if abs_href == target_url_to_boost or href == target_path:
                                    has_link = True
                                    anchor = a.get_text(strip=True).lower()
                                    if anchor in normalized_anchors:
                                        is_exact_match_link = True
                                        break
                                    else:
                                        link_anchor = anchor
                            
                            if is_exact_match_link:
                                continue
                            
                            elif has_link:
                                reviews.append({
                                    'Page URL': current_url,
                                    'Existing Link Anchor': link_anchor,
                                    'Action': 'Review: Change anchor?'
                                })
                                
                            else:
                                # Clean further for text analysis
                                for tag in soup(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'a']):
                                    tag.decompose()
                                
                                text = soup.get_text(separator=' ', strip=True)[:15000]
                                if not text: continue
                                
                                matches = [k for k in possible_anchors if k.lower() in text.lower()]
                                
                                score = 0
                                try:
                                    vec = get_embedding(text, client)
                                    if vec:
                                        score = cosine_similarity([target_vec], [vec])[0][0]
                                except: pass
                                
                                if matches or score > 0.73:
                                    opportunities.append({
                                        'Page URL': current_url,
                                        'Target URL': target_url_to_boost,
                                        'Keywords Found': ", ".join(matches),
                                        'Semantic Score': round(score, 4),
                                        'Type': 'Keyword' if matches else 'Semantic'
                                    })
                        
                        progress_bar.empty()
                        
                        # --- Display Results ---
                        
                        # 1. New Opportunities
                        st.subheader("🚀 New Linking Opportunities")
                        if opportunities:
                            df_ops = pd.DataFrame(opportunities).sort_values('Semantic Score', ascending=False)
                            st.dataframe(df_ops, use_container_width=True)
                        else:
                            st.info("No new opportunities found.")

                        # 2. Review Existing Links
                        st.subheader("🔍 Review Existing Links")
                        if reviews:
                            df_rev = pd.DataFrame(reviews)
                            st.dataframe(df_rev, use_container_width=True)
                        else:
                            st.info("No existing links found that need review.")

                        # 3. Export
                        if opportunities or reviews:
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                if opportunities:
                                    df_ops.to_excel(writer, sheet_name='New Opportunities', index=False)
                                if reviews:
                                    df_rev.to_excel(writer, sheet_name='Existing Link Review', index=False)
                            
                            st.download_button(
                                label="Download Results as Excel",
                                data=output.getvalue(),
                                file_name="internal_link_opportunities.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                    else:
                        st.error("Failed to generate embedding for target page.")
        else:
            st.error("No pages found to analyze.")
