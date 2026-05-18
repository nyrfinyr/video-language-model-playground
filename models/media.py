from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Text:
    type: str = field(default="text", init=False)
    text: str


@dataclass(frozen=True)
class Image:
    type: str = field(default="image", init=False)
    image: str


@dataclass(frozen=True)
class Video:
    type: str = field(default="video", init=False)
    video: str | Path
    max_pixels: int = 360 * 420
    fps: float = 0.5


@dataclass(frozen=True)
class VideoFrames:
    type: str = field(default="video", init=False)
    video: list[str]


MediaItem = Image | Video | VideoFrames
