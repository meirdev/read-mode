import argparse

import requests
from bs4 import BeautifulSoup
from readability import Document
from transformers import pipeline

MODEL = "facebook/bart-large-cnn"


def get_summary(url: str, min_length: int, max_length: int) -> str | None:
    response = requests.get(url)
    response.raise_for_status()

    doc = Document(response.content)
    title = doc.title()
    summary = doc.summary()

    soup = BeautifulSoup(summary, features="lxml")
    text = soup.get_text()

    full_text = f"{title}\n\n{text}"

    summarizer = pipeline("summarization", model=MODEL)
    summary_text = summarizer(full_text, min_length=min_length, max_length=max_length)

    if len(summary_text):
        return summary_text[0]["summary_text"]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a web page")
    parser.add_argument("url", help="URL of the web page to summarize")
    parser.add_argument("--min-length", type=int, default=50, help="Minimum length of the summary")
    parser.add_argument("--max-length", type=int, default=250, help="Maximum length of the summary")

    args = parser.parse_args()

    summary = get_summary(args.url, args.min_length, args.max_length)

    if summary:
        print(summary)
    else:
        print("Unable to summarize")


if __name__ == "__main__":
    main()
