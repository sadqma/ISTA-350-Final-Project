"""
scraper.py

ISTA 350 Final Project
Web scraping of two web-based datasets from Wikipedia:

1) Bite force of different animals
2) Deadliest animals (estimated deaths per year)

This script downloads HTML tables from the web, converts them to CSV
files, and saves them into the local data/ directory.

Head developer: Danel Khamit
"""

import os
import requests
import pandas as pd


BITE_FORCE_URL = "https://en.wikipedia.org/wiki/Bite_force"
DEADLIEST_ANIMALS_URL = "https://en.wikipedia.org/wiki/List_of_animals_deadliest_to_humans"


def ensure_data_dir() -> str:
    """Create data/ directory if it doesn't exist and return its path."""
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def scrape_table(url: str) -> pd.DataFrame:
    """
    Scrape the FIRST HTML table from a given URL.

    1) Делает HTTP-запрос с нормальным User-Agent.
    2) Берёт HTML и парсит таблицы через pandas.read_html.

    Parameters
    ----------
    url : str
        Web address of the page that contains an HTML table.

    Returns
    -------
    df : pandas.DataFrame
        The first table found on the page.
    """
    print(f"Scraping tables from: {url}")

    # Делаем запрос как "обычный браузер", чтобы не получать 403
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/119.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()  # если ошибка HTTP — выбросит исключение

    # Парсим HTML, который мы уже скачали
    tables = pd.read_html(resp.text)
    if not tables:
        raise ValueError(f"No tables found at {url}")

    df = tables[0]
    print(f"  Found table with shape {df.shape}")
    return df


def save_csv(df: pd.DataFrame, path: str) -> None:
    """Save a DataFrame as CSV to the given path."""
    df.to_csv(path, index=False)
    print(f"  Saved CSV to {path}")


def main() -> None:
    """Run all scraping tasks."""
    data_dir = ensure_data_dir()

    # 1. Bite force dataset
    try:
        bite_df = scrape_table(BITE_FORCE_URL)
        bite_path = os.path.join(data_dir, "bite_force_animals.csv")
        save_csv(bite_df, bite_path)
    except Exception as e:
        print(f"Error scraping bite force data: {e}")

    # 2. Deadliest animals dataset
    try:
        deadliest_df = scrape_table(DEADLIEST_ANIMALS_URL)
        deadliest_path = os.path.join(data_dir, "deadliest_animals_to_humans.csv")
        save_csv(deadliest_df, deadliest_path)
    except Exception as e:
        print(f"Error scraping deadliest animals data: {e}")


if __name__ == "__main__":
    main()
