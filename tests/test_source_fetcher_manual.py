from aca.tools.source_fetcher import get_source_fetcher


def main() -> None:
    fetcher = get_source_fetcher()

    sources = {
        "thermo": "https://www.thermofisher.com/",
        "olink": "https://olink.com/",
    }

    pages = fetcher.fetch_pages(sources)

    for source_id, page in pages.items():
        print()
        print("=" * 80)
        print(f"SOURCE: {source_id}")
        print(f"URL: {page.url}")
        print(f"SUCCESS: {page.success}")

        if page.error:
            print(f"ERROR: {page.error}")

        print(f"CONTENT LENGTH: {len(page.content)}")

        if page.content:
            print()
            print(page.content[:1000])


if __name__ == "__main__":
    main()