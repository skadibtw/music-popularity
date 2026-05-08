"""Write the popular-reference feature research report from current artifacts."""

try:
    from src.research_insights import PROJECT_ROOT, build_research_summary, format_research_markdown
except ModuleNotFoundError:
    from research_insights import PROJECT_ROOT, build_research_summary, format_research_markdown


OUTPUT_PATH = PROJECT_ROOT / "reports/popular_feature_research.md"


def main() -> None:
    summary = build_research_summary()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(format_research_markdown(summary), encoding="utf-8")
    print(f"Research report saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
