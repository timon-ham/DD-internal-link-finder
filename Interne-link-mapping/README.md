# 🔗 Internal Link Opportunity Finder

This tool crawls a website and suggests internal linking opportunities based on **keyword matching** and **semantic similarity**.

## Prerequisites

- **Python 3.8+** installed on your system.
- An **OpenAI API Key** (for semantic analysis).

## Installation

1.  **Download the files**: Ensure you have `streamlit_app.py` and `requirements.txt` in a folder.
2.  **Open a terminal**: Navigate to the folder where you saved the files.
    - *Windows Tip*: Shift + Right-click in the folder and select "Open PowerShell window here" or "Open in Terminal".
3.  **Install Dependencies**: Run the following command:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

1.  In the terminal, run:
    ```bash
    streamlit run streamlit_app.py
    ```
2.  A new tab will open in your default web browser with the app.

## Using the App

1.  **API Key**: Enter your OpenAI API Key in the sidebar (left panel).
2.  **Configuration**: Optionally adjust "Max Pages" and "Crawl Delay" in the sidebar.
3.  **Inputs**:
    - **Root URL**: The homepage of the site you want to crawl (e.g., `https://example.com`).
    - **Target URL**: The specific page you want to build links *to* (e.g., `https://example.com/important-page`).
    - **Anchor Texts**: Comma-separated list of keywords you want to find (e.g., `seo services, link building, audit`).
4.  **Start**: Click the "Start Analysis" button.

## Output

- The tool will display **New Opportunities** (pages that mention your keywords or are semantically relevant).
- It will also show **Existing Links** that might need their anchor text optimized.
- You can **download the results** as an Excel file using the download button at the bottom.
