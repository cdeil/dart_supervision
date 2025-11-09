// Example how to load and work with real detections data and annotate images.
import 'dart:io';
import 'package:image/image.dart' as img;
import 'package:dart_supervision/dart_supervision.dart';

void main() async {
  print('Loading detections from JSON...');

  // Load detection data for frame 0
  final detections = await DetectionUtils.loadFromJson(
    'assets/people-walking-detections.json',
    0,
  );

  // Get video info
  final videoInfo = await DetectionUtils.getVideoInfo(
    'assets/people-walking-detections.json',
  );

  print('Video info: ${videoInfo['video_info']}');
  print('Model info: ${videoInfo['model_info']}');
  print('Frame 0 detections: ${detections.length}');

  // Create annotators
  final boxAnnotator = BoxAnnotator(
    color: img.ColorRgb8(0, 255, 0), // Green boxes
    thickness: 3,
  );
  final labelAnnotator = LabelAnnotator(
    textColor: img.ColorRgb8(255, 255, 255), // White text
    backgroundColor: img.ColorRgb8(0, 0, 0), // Black background
    fontSize: 12,
    padding: 2,
  );

  // Process all frames and save annotated images
  print('Processing frames...');
  
  final allDetections = await DetectionUtils.loadAllFramesFromJson(
    'assets/people-walking-detections.json',
  );

  // Annotate a few example frames
  final framesToProcess = [0, 50, 100, 150]; // Sample frames
  for (final frameIndex in framesToProcess) {
    if (frameIndex >= allDetections.length) continue;
    
    final frameDetections = allDetections[frameIndex];
    
    // Read the PNG file (assuming we have frame images)
    final pngBytes = await File('assets/people-walking.png').readAsBytes();
    final pngImage = img.decodePng(pngBytes);

    if (pngImage != null) {
      // Annotate with bounding boxes and labels
      boxAnnotator.annotate(pngImage, frameDetections);
      labelAnnotator.annotate(pngImage, frameDetections);

      // Save annotated frame
      final encodedPng = img.encodePng(pngImage);
      await File('assets/people-walking-frame-$frameIndex-annotated.png')
          .writeAsBytes(encodedPng);
      print('Saved annotated frame $frameIndex');
    }
  }

  // Show detection details for first frame
  print('\nDetection details for frame 0:');
  for (int i = 0; i < detections.length && i < 5; i++) {
    final x1 = detections.xyxy[[i, 0]];
    final y1 = detections.xyxy[[i, 1]];
    final x2 = detections.xyxy[[i, 2]];
    final y2 = detections.xyxy[[i, 3]];

    var info = 'Detection $i: bbox=[${x1.toStringAsFixed(1)}, ${y1.toStringAsFixed(1)}, ${x2.toStringAsFixed(1)}, ${y2.toStringAsFixed(1)}]';

    if (detections.confidence != null) {
      final conf = detections.confidence![[i]];
      info += ', conf=${conf.toStringAsFixed(3)}';
    }

    if (detections.trackerId != null) {
      final tid = detections.trackerId![[i]].round();
      info += ', track_id=$tid';
    }

    print(info);
  }

  // Print summary statistics
  final totalDetections = allDetections.fold<int>(0, (sum, d) => sum + d.length);
  final avgDetections = totalDetections / allDetections.length;

  // Get unique tracker IDs
  final uniqueTrackers = <int>{};
  for (final frameDetections in allDetections) {
    if (frameDetections.trackerId != null) {
      for (int i = 0; i < frameDetections.length; i++) {
        uniqueTrackers.add(frameDetections.trackerId![[i]].round());
      }
    }
  }

  print('\nOverall Statistics:');
  print('- Total frames: ${allDetections.length}');
  print('- Total detections: $totalDetections');
  print('- Average detections per frame: ${avgDetections.toStringAsFixed(1)}');
  print('- Unique tracks: ${uniqueTrackers.length}');

  print('\nDone! Check the annotated images in assets/');
}
