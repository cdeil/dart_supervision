#!/usr/bin/env python3
import json
from pathlib import Path
from ultralytics.models import YOLO
import supervision as sv


def normalize_detections(
    detections: sv.Detections, frame_width: int, frame_height: int
) -> dict:
    """Convert detections to normalized coordinates (0-1) for any video size."""
    result = {
        "xyxy": [],
        "confidence": [],
        "class_id": [],
        "tracker_id": [],
    }

    if len(detections) == 0:
        return result

    # Normalize bounding boxes to 0-1 range
    normalized_xyxy = detections.xyxy.copy()
    normalized_xyxy[:, [0, 2]] /= frame_width  # x coordinates
    normalized_xyxy[:, [1, 3]] /= frame_height  # y coordinates

    result["xyxy"] = normalized_xyxy.tolist()
    result["confidence"] = (
        detections.confidence.tolist()
        if detections.confidence is not None
        else [1.0] * len(detections)
    )
    result["class_id"] = (
        detections.class_id.tolist()
        if detections.class_id is not None
        else [0] * len(detections)
    )
    result["tracker_id"] = (
        detections.tracker_id.tolist()
        if detections.tracker_id is not None
        else list(range(len(detections)))
    )

    return result


def main():
    # Paths
    video_path = Path("assets/people-walking.mp4")
    output_path = Path("assets/people-walking-detections.json")

    if not video_path.exists():
        print(f"Error: Video file {video_path} not found")
        return

    # Load YOLO model and initialize tracker
    print("Loading YOLO11n model...")
    model = YOLO("yolo11n.pt")
    tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=30,
    )

    # Get video info
    video_info = sv.VideoInfo.from_video_path(str(video_path))
    print(
        f"Video info: {video_info.width}x{video_info.height}, {video_info.fps} fps, {video_info.total_frames} frames"
    )

    # Process video and collect results
    results = []

    for frame_idx, frame in enumerate(sv.get_video_frames_generator(str(video_path))):
        # Run YOLO detection
        yolo_results = model(frame, verbose=False)
        detections = sv.Detections.from_ultralytics(yolo_results[0])

        # Filter for person class (class_id = 0 in COCO)
        if detections.class_id is not None:
            detections = detections[detections.class_id == 0]

        # Apply tracking
        detections = tracker.update_with_detections(detections)

        # Convert to normalized coordinates and store
        normalized = normalize_detections(
            detections, video_info.width, video_info.height
        )
        results.append(
            {
                "frame_index": frame_idx,
                "timestamp": frame_idx / (video_info.fps or 30),
                "detections": normalized,
            }
        )

        if frame_idx % 30 == 0:  # Print progress every 30 frames
            print(
                f"Processed frame {frame_idx}/{video_info.total_frames or 'unknown'} - {len(detections)} detections"
            )

    # Create final JSON structure
    output_data = {
        "video_info": {
            "width": video_info.width,
            "height": video_info.height,
            "fps": video_info.fps or 30,
            "total_frames": video_info.total_frames or len(results),
            "duration_seconds": len(results) / (video_info.fps or 30),
        },
        "model_info": {
            "model_name": "yolo11n",
            "classes_detected": ["person"],
            "tracking_enabled": True,
        },
        "frames": results,
    }

    # Save to JSON
    print(f"Saving {len(results)} frames to {output_path}")
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
