#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Recursive pagination for Eastmoney stock news
"""

import json
import logging
import time
from datetime import datetime

import pandas as pd
import requests

log = logging.getLogger(__name__)


class EastmoneyFetcher:
    """
    A class to fetch data from Eastmoney.
    """

    def fetch_news_for_symbol(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch Eastmoney news for a stock within a date range and standardize columns.
        Dates: 'YYYY-MM-DD'
        """
        log.info(f"Fetching news for {symbol} from {start_date} to {end_date} using Eastmoney...")
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date).replace(hour=23, minute=59, second=59)

        try:
            df = self._stock_news_em_all(
                symbol=symbol.split('.')[0],
                start_date=start_date,
                end_date=end_date,
                page_size=100,
                max_pages=50
            )
            if df.empty:
                log.info(f"Eastmoney returned no news for symbol {symbol}.")
                return pd.DataFrame()

            # --- Standardize Column Names ---
            rename_map = {
                'date': 'datetime',
                'title': 'title',
                'content': 'content',
                'art_url': 'url',
                'Url': 'url'
            }
            actual_rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
            if actual_rename_map:
                df.rename(columns=actual_rename_map, inplace=True)

            # --- Validate Essential Columns ---
            essential_cols = ['datetime', 'title']
            if not all(col in df.columns for col in essential_cols):
                log.error(
                    f"Could not find all essential columns {essential_cols} "
                    f"for {symbol}. Available: {df.columns.tolist()}"
                )
                return pd.DataFrame()

            # --- Ensure 'content' and 'url' columns exist ---
            if 'content' not in df.columns:
                log.warning(f"Column 'content' not found in fetched data for {symbol}. Filling with None.")
                df['content'] = None
            if 'url' not in df.columns:
                log.warning(f"Column 'url' not found in fetched data for {symbol}. Filling with None.")
                df['url'] = None

            # Convert datetime column and handle potential errors
            df['datetime'] = pd.to_datetime(df['datetime'], errors="coerce")
            df.dropna(subset=['datetime'], inplace=True)

            if not df.empty:
                log.info(f"Eastmoney API returned {len(df)} unique raw articles for {symbol}. "
                         f"Their date range is from {df['datetime'].min()} to {df['datetime'].max()}. "
                         f"Requested range was {start_date} to {end_date}.")
            else:
                log.info(f"Eastmoney returned no valid news in the date range for {symbol}.")

            # Local filtering is necessary because the API does not strictly respect the date range.
            mask = (df['datetime'] >= start_dt) & (df['datetime'] <= end_dt)
            filtered_df = df[mask]
            
            log.info(f"Successfully filtered to {len(filtered_df)} articles for {symbol} within the requested range.")
            return filtered_df.copy()

        except Exception as e:
            log.error(f"An unexpected error occurred fetching news for {symbol} from Eastmoney: {e}", exc_info=True)
            return pd.DataFrame()

    def _stock_news_em_all(
        self,
        symbol: str,
        start_date: str = None,
        end_date: str = None,
        page_size: int = 100,
        max_pages: int = 10,
        delay: float = 0.5,
    ) -> pd.DataFrame:
        """
        东方财富-个股新闻-分页获取全部新闻, 支持日期范围.
        https://so.eastmoney.com/news/s?keyword=603777
        """
        url = "http://search-api-web.eastmoney.com/search/jsonp"
        all_dfs = []
        max_retries = 3
        retry_sleep = 10

        for page in range(1, max_pages + 1):
            cms_params = {
                "searchScope": "default",
                "sort": "default",
                "pageIndex": page,
                "pageSize": page_size,
                "preTag": "<em>",
                "postTag": "</em>"
            }
            if start_date:
                cms_params["beginTime"] = f"{start_date} 00:00:00"
            if end_date:
                cms_params["endTime"] = f"{end_date} 23:59:59"

            params = {
                "cb": "jQuery3510875346244069884_1668256937995",
                "param": json.dumps({
                    "uid": "",
                    "keyword": symbol,
                    "type": ["cmsArticle"],
                    "client": "web",
                    "clientType": "web",
                    "clientVersion": "curr",
                    "param": {
                        "cmsArticle": cms_params
                    }
                }, ensure_ascii=False)
            }

            r = None
            for attempt in range(max_retries):
                try:
                    r = requests.get(url, params=params, timeout=20)
                    r.raise_for_status()
                    break
                except requests.exceptions.RequestException as e:
                    log.warning(f"Attempt {attempt + 1}/{max_retries} failed for page {page} of {symbol}. Error: {e}")
                    if attempt + 1 < max_retries:
                        log.info(f"Sleeping for {retry_sleep} seconds before retrying.")
                        time.sleep(retry_sleep)
                    else:
                        log.error(f"All retries failed for page {page} of {symbol}. Skipping page.")
            
            if not r:
                continue

            data_text = r.text

            try:
                start_index = data_text.find('(')
                end_index = data_text.rfind(')')
                if start_index == -1 or end_index == -1:
                    log.info(f"No more results at page {page}, or malformed JSONP.")
                    break

                json_str = data_text[start_index + 1:end_index]
                data_json = json.loads(json_str)
            except json.JSONDecodeError:
                log.error(f"Failed to decode JSON from response for page {page}: {data_text}")
                continue

            articles = data_json.get("result", {}).get("cmsArticle", [])
            if not articles:
                log.info(f"No more results at page {page}")
                break

            temp_df = pd.DataFrame(articles)
            log.info(f"Fetched page {page}, got {len(temp_df)} items with date range: {temp_df['date'].min()} to {temp_df['date'].max()}")
            all_dfs.append(temp_df)

            time.sleep(delay)

        if all_dfs:
            full_df = pd.concat(all_dfs, ignore_index=True)
            initial_rows = len(full_df)

            # Normalize content to handle variations in <em> tags for robust de-duplication.
            if 'content' in full_df.columns:
                full_df['normalized_content'] = full_df['content'].str.replace('<em>', '', regex=False).str.replace('</em>', '', regex=False)
                subset_cols = ['date', 'title', 'normalized_content']
            else:
                subset_cols = ['date', 'title']

            full_df.drop_duplicates(subset=subset_cols, inplace=True)
            
            # Clean up the temporary column if it was created
            if 'normalized_content' in full_df.columns:
                full_df.drop(columns=['normalized_content'], inplace=True)

            final_rows = len(full_df)
            if initial_rows > final_rows:
                log.info(f"Removed {initial_rows - final_rows} duplicate articles found across pages.")
            else:
                log.info("No duplicate articles found across pages.")

            return full_df
        else:
            return pd.DataFrame()
