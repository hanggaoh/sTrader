#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Recursive pagination for Eastmoney stock news with date filtering.
"""

import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("data.eastmoney_fetcher")
from data.eastmoney_fetcher import EastmoneyFetcher
def main():
    logging.basicConfig(level=logging.INFO)
    fetcher = EastmoneyFetcher()
    df = fetcher.fetch_news_for_symbol("603777.SS", "2025-01-01", "2025-02-01")
    print(df.head())
    print(f"Total fetched: {len(df)} rows")

if __name__ == "__main__":
    main()
