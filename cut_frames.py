"""Cut frames from videos to build the real target-domain image set."""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
from tqdm import tqdm


VIDEO_EXTS: Tuple[str, ...] = (
    '.mp4', '.MP4', '.mov', '.MOV',
    '.avi', '.AVI', '.mkv', '.MKV',
)


@dataclass(frozen=True)
class FrameCutterConfig:
    fps_target: float = 1.0
    max_frames_per_video: int = 0
    img_ext: str = '.jpg'
    quality: int = 95

    def write_params(self) -> List[int]:
        ext = self.img_ext.lower()
        if ext in ('.jpg', '.jpeg'):
            return [cv2.IMWRITE_JPEG_QUALITY, int(self.quality)]
        if ext == '.png':
            return [cv2.IMWRITE_PNG_COMPRESSION, 3]
        return []


class VideoFrameCutter:
    """Decode + sample frames from one video at a target FPS."""

    def __init__(self, config: FrameCutterConfig) -> None:
        self.config = config

    def cut(
        self,
        video_path: str,
        out_dir: str,
        outer_bar: Optional[tqdm] = None,
    ) -> int:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            tqdm.write(f'  [WARN] cannot open {video_path}')
            return 0

        cfg = self.config
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        step = max(1, int(round(fps_src / max(cfg.fps_target, 1e-6))))
        stem = Path(video_path).stem
        write_params = cfg.write_params()

        idx = 0
        saved = 0
        bar = tqdm(
            total=total or None,
            desc=stem,
            unit='f',
            leave=False,
            colour='red',
            dynamic_ncols=True,
        )
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % step == 0:
                    fname = f'{stem}_{idx:06d}{cfg.img_ext}'
                    cv2.imwrite(
                        os.path.join(out_dir, fname),
                        frame,
                        write_params,
                    )
                    saved += 1
                    bar.set_postfix(saved=saved, refresh=False)
                    if cfg.max_frames_per_video and saved >= cfg.max_frames_per_video:
                        break
                idx += 1
                bar.update(1)
        finally:
            bar.close()
            cap.release()

        if outer_bar is not None:
            outer_bar.set_postfix(last=stem, frames=saved, refresh=False)
        return saved


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Cut frames from videos for the target domain.'
    )
    p.add_argument('--path-input', required=True, type=str,
                   help='Folder containing source videos.')
    p.add_argument('--path-output', required=True, type=str,
                   help='Folder where extracted frames will be saved.')
    p.add_argument('--fps', type=float, default=1.0,
                   help='Target sampling rate (frames/second).')
    p.add_argument('--max-frames-per-video', type=int, default=0,
                   help='Cap on frames saved per video (0 = no cap).')
    p.add_argument('--img-ext', type=str, default='.jpg')
    p.add_argument('--quality', type=int, default=95,
                   help='JPEG quality 1-100.')
    return p.parse_args()


def _list_videos(folder: str) -> List[str]:
    return sorted(f for f in os.listdir(folder) if f.endswith(VIDEO_EXTS))


def main() -> None:
    args = _parse_args()

    if not os.path.isdir(args.path_input):
        raise FileNotFoundError(
            f'path-input does not exist: {args.path_input}'
        )
    os.makedirs(args.path_output, exist_ok=True)

    videos = _list_videos(args.path_input)
    if not videos:
        print(
            f'No videos with extensions {VIDEO_EXTS} found in '
            f'{args.path_input}'
        )
        return

    print(
        f'Found {len(videos)} videos. Sampling at ~{args.fps} fps -> '
        f'{args.path_output}'
    )

    cutter = VideoFrameCutter(FrameCutterConfig(
        fps_target=args.fps,
        max_frames_per_video=args.max_frames_per_video,
        img_ext=args.img_ext,
        quality=args.quality,
    ))

    total = 0
    outer = tqdm(
        videos, desc='videos', unit='vid',
        colour='red', dynamic_ncols=True,
    )
    for v in outer:
        n = cutter.cut(
            os.path.join(args.path_input, v),
            args.path_output,
            outer_bar=outer,
        )
        total += n
    outer.close()
    print(f'Done. Total frames saved: {total}')


if __name__ == '__main__':
    main()
