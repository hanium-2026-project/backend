"""촬영 영상에서 학습용 프레임을 추출한다.

목표 장수를 주면 영상을 그만큼의 구간으로 나누고, 각 구간에서 가장 선명한
프레임을 하나씩 고른다. 시간 분포를 고르게 유지하면서 모션 블러를 피할 수 있고,
영상 특성과 무관하게 항상 목표 장수를 채운다 (절대 선명도 임계값은 영상마다
기준이 달라 신뢰하기 어렵다).

여러 영상을 한 번에 처리하며, 영상별 접두어를 붙여 파일명이 겹치지 않게 한다.

사용법::

    # 전체 300장을 영상 4개에 고르게 배분
    python tools/extract_frames.py videos/*.mov -o frames/ --total 300

    # 영상마다 80장씩
    python tools/extract_frames.py a.mov b.mov -o frames/ --per-video 80

    # 구간당 후보를 늘려 더 선명한 프레임 선택 (느려짐)
    python tools/extract_frames.py a.mov -o frames/ --total 200 --candidates 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def blur_score(image) -> float:
    """라플라시안 분산 — 낮을수록 흐리다 (모션 블러 프레임 제외용)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract(video: Path, out_dir: Path, count: int,
            candidates: int, prefix: str) -> int:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"  [건너뜀] 열 수 없음: {video}")
        return 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if total <= 0:
        print(f"  [건너뜀] 프레임 수를 읽을 수 없음: {video}")
        cap.release()
        return 0

    # 영상을 count 개 구간으로 나누고, 각 구간의 후보 중 가장 선명한 것을 고른다
    saved = 0
    scores: list[float] = []
    for slot in range(count):
        lo = int(total * slot / count)
        hi = max(lo + 1, int(total * (slot + 1) / count))
        best_frame = None
        best_score = -1.0
        for k in range(candidates):
            idx = lo + (hi - lo) * k // max(candidates, 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(idx, total - 1))
            ok, frame = cap.read()
            if not ok:
                continue
            score = blur_score(frame)
            if score > best_score:
                best_score, best_frame = score, frame
        if best_frame is None:
            continue
        cv2.imwrite(str(out_dir / f"{prefix}_{saved:04d}.jpg"), best_frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        scores.append(best_score)
        saved += 1

    cap.release()
    dur = total / fps
    sharp = f", 선명도 중앙값 {sorted(scores)[len(scores) // 2]:.0f}" if scores else ""
    print(f"  {video.name}: {saved}장 저장 ({dur:.1f}초, {total}프레임{sharp})")
    return saved


def main() -> int:
    ap = argparse.ArgumentParser(description="영상에서 학습용 프레임 추출")
    ap.add_argument("videos", nargs="+", help="영상 파일 경로들")
    ap.add_argument("-o", "--out", default="frames", help="출력 디렉토리")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--total", type=int, help="전체 목표 장수 (영상들에 균등 배분)")
    group.add_argument("--per-video", type=int, help="영상당 장수")
    ap.add_argument("--candidates", type=int, default=3,
                    help="구간당 비교할 후보 프레임 수 (많을수록 선명하지만 느림)")
    args = ap.parse_args()

    videos = [Path(v) for v in args.videos if Path(v).exists()]
    if not videos:
        print("처리할 영상이 없습니다.")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    per = args.per_video or max(1, args.total // len(videos))
    print(f"영상 {len(videos)}개 → {out_dir}/ (영상당 {per}장 목표)\n")

    total_saved = 0
    for i, video in enumerate(videos):
        total_saved += extract(video, out_dir, per, args.candidates, f"v{i + 1}")

    print(f"\n총 {total_saved}장 추출 완료 → {out_dir}/")
    if total_saved < per * len(videos):
        print("일부 구간에서 프레임을 읽지 못했습니다. 영상 파일을 확인하세요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
