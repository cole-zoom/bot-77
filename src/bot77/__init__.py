import argparse
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="bot77")
    sub = parser.add_subparsers(dest="command", required=True)

    cards = sub.add_parser("cards", help="card metadata")
    cards_sub = cards.add_subparsers(dest="cards_command", required=True)
    ingest = cards_sub.add_parser("ingest", help="merge API snapshot + curated facts into LanceDB")
    ingest.add_argument("--api-json", type=Path, help="defaults to the newest file in data/cards/api_snapshots")
    ingest.add_argument("--curated", type=Path, default=Path("data/cards/curated.yaml"))
    ingest.add_argument("--db", type=Path, default=Path("data/lancedb"))
    ingest.add_argument("--art-dir", type=Path, default=Path("data/cards/art"))

    calib = sub.add_parser("calibrate", help="draw a layout's HUD regions on random frames")
    calib.add_argument("--layout", required=True, help="layout id in config/layouts")
    calib.add_argument("--video", type=Path, required=True)
    calib.add_argument("-n", type=int, default=10)
    calib.add_argument("--seed", type=int, default=0)
    calib.add_argument("--out", type=Path, help="defaults to data/calibration/<layout>_<video>.jpg")

    proc = sub.add_parser("process", help="video -> matches, HUD states and play events in LanceDB")
    proc.add_argument("--video", type=Path, required=True)
    proc.add_argument("--layout", required=True)
    proc.add_argument("--creator", default="")
    proc.add_argument("--mode", default="ladder", help="game mode; clock rules are only known for ladder")
    proc.add_argument("--db", type=Path, default=Path("data/lancedb"))

    review = sub.add_parser("review", help="write an HTML page for checking a processed video's events")
    review.add_argument("--video-id", required=True)
    review.add_argument("--db", type=Path, default=Path("data/lancedb"))
    review.add_argument("--out", type=Path, help="defaults to data/reviews/<video-id>.html")

    score = sub.add_parser("score", help="precision/recall of detected plays from an exported review")
    score.add_argument("review_json", type=Path)
    score.add_argument("--db", type=Path, default=Path("data/lancedb"))

    args = parser.parse_args()
    if args.command == "cards" and args.cards_command == "ingest":
        _cards_ingest(args)
    elif args.command == "calibrate":
        _calibrate(args)
    elif args.command == "process":
        _process(args)
    elif args.command == "review":
        _review(args)
    elif args.command == "score":
        import json

        from bot77.review import score_review

        print(json.dumps(score_review(args.db, args.review_json), indent=2))


def _process(args) -> None:
    import lancedb

    from bot77.pipeline import process_video

    cards_table = lancedb.connect(args.db).open_table("cards")
    cards = cards_table.search().where("kind = 'deck_card'").limit(1000).to_arrow().to_pylist()
    snapshot = cards[0]["snapshot"] if cards else ""
    result = process_video(args.video, args.layout, cards, args.db, creator=args.creator, mode=args.mode,
                           card_snapshot=snapshot)
    print(f"Done: {result}")


def _review(args) -> None:
    from bot77.review import write_review

    out = args.out or Path("data/reviews") / f"{args.video_id}.html"
    n = write_review(args.db, args.video_id, out)
    print(f"Wrote {out} ({n} events)")


def _calibrate(args) -> None:
    from bot77.calibrate import overlay_sheet, region_strips
    from bot77.layout import Layout

    out = args.out or Path("data/calibration") / f"{args.layout}_{args.video.stem}.jpg"
    layout = Layout.load(args.layout)
    times = overlay_sheet(args.video, layout, out, n=args.n, seed=args.seed)
    strips = out.with_name(out.stem + "_regions.png")
    region_strips(args.video, layout, strips, times)
    print(f"Wrote {out} and {strips} ({len(times)} frames at {', '.join(f'{t:.0f}s' for t in times)})")


def _cards_ingest(args) -> None:
    from bot77.cards import ingest

    api_json = args.api_json or max(Path("data/cards/api_snapshots").glob("*.json"))
    report = ingest(api_json, args.curated, args.db, args.art_dir)

    kinds = Counter(r["kind"] for r in report.rows)
    print(f"Wrote {len(report.rows)} rows to {args.db}/cards from {api_json.name}: {dict(kinds)}")
    if report.warnings:
        # Group the "not filled in yet" warnings so they don't drown out real problems.
        todo = [w for w in report.warnings if "not filled in" in w]
        other = [w for w in report.warnings if w not in todo]
        for w in other:
            print(f"  WARNING {w}")
        if todo:
            print(f"  {len(todo)} curated fields still to fill in (evo cycles / abilities)")
