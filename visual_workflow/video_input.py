"""Provider constraints for optional direct-video model input."""

from pathlib import Path


# MIME types documented for Gemini generateContent video understanding.
GEMINI_VIDEO_MIME_TYPES = {
    ".mp4": "video/mp4",
    ".mpeg": "video/mpeg",
    ".mpe": "video/mpeg",
    ".mov": "video/quicktime",
    ".avi": "video/avi",
    ".flv": "video/x-flv",
    ".mpg": "video/mpg",
    ".webm": "video/webm",
    ".wmv": "video/wmv",
    ".3gp": "video/3gpp",
    ".3gpp": "video/3gpp",
}


def gemini_supports_inline_video(video_path) -> bool:
    return Path(video_path).suffix.lower() in GEMINI_VIDEO_MIME_TYPES


def gemini_video_mime_type(video_path) -> str:
    extension = Path(video_path).suffix.lower()
    try:
        return GEMINI_VIDEO_MIME_TYPES[extension]
    except KeyError as ex:
        raise ValueError(
            f"Gemini does not list {extension or '<no extension>'} as a supported "
            "inline video format; convert the file to MP4 first"
        ) from ex
