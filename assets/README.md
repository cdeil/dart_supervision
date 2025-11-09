# Video Assets

This directory contains the people walking video in multiple animated formats for testing the dart_supervision library.

## Files

### Original Assets

- **people-walking.mp4** (7.3M) - Original video (1920x1080, 25 fps, 341 frames)
- **people-walking.gif** (9.9M) - Animated GIF (640x360, 5 fps, 68 frames)
- **people-walking.webp** (2.3M) - Animated WEBP (640x360, 10 fps, 136 frames)
- **people-walking.png** (6.7M) - Animated PNG/APNG (640x360, 5 fps, 68 frames, grayscale)

### Annotated Assets (with red rectangle)

- **people-walking-annotated.gif** (10.5M) - GIF with red rectangle on all 68 frames
- **people-walking-annotated.png** (6.5M) - PNG with red rectangle (single frame)
- **people-walking-annotated-from-webp.png** (20.7M) - WEBP converted to PNG with red rectangle

## FFmpeg Commands Used

```bash
# Download original
curl -o people-walking.mp4 "https://media.roboflow.com/supervision/video-examples/people-walking.mp4"

# Create animated GIF (5 fps, every second frame)
ffmpeg -i people-walking.mp4 -vf "fps=5,scale=640:-1:flags=lanczos,palettegen=reserve_transparent=0" -y palette.png
ffmpeg -i people-walking.mp4 -i palette.png -lavfi "fps=5,scale=640:-1:flags=lanczos [x]; [x][1:v] paletteuse" -y people-walking.gif

# Create animated WEBP (10 fps)
ffmpeg -i people-walking.mp4 -vf "fps=10,scale=640:-1" -c:v libwebp -quality 80 -compression_level 6 -preset default -loop 0 -y people-walking.webp

# Create animated PNG/APNG (5 fps, grayscale for smaller size)
ffmpeg -i people-walking.mp4 -vf "fps=5,scale=640:-1,format=gray" -f apng -y people-walking.png
```

## Annotation Example

Run the minimal annotation example to demonstrate read/write capabilities:

```bash
dart run assets/annotate_example.dart
```

This simple script shows:

- ✅ **Animated GIF**: Reads all 68 frames, draws red rectangle on each, writes annotated GIF
- ✅ **PNG (APNG)**: Reads first frame, draws rectangle, writes PNG
- ⚠️ **WEBP**: Reads first frame, draws rectangle, saves as PNG (WEBP encoding not supported)

## Testing

Basic image format reading can be tested:

```bash
dart test test/animated_image_test.dart
```
