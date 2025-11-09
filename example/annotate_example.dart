// Simplified example: read one image and annotate all frames in place.
import 'dart:io';
import 'package:image/image.dart' as img;
import 'package:dart_supervision/dart_supervision.dart' as sv;

void main() async {
  print('Loading image and detections...');

  // TODO: change back to PNG, then fix it

  // Read the GIF image
  final gifBytes = await File('assets/people-walking.gif').readAsBytes();
  final image = img.decodeGif(gifBytes);

  // Load all detection data
  final allDetections = await sv.DetectionUtils.loadAllFramesFromJson(
    'assets/people-walking-detections.json',
  );

  // Create annotators
  final boxAnnotator =
      sv.BoxAnnotator(color: img.ColorRgb8(255, 255, 255), thickness: 3);
  final labelAnnotator = sv.LabelAnnotator(
      textColor: img.ColorRgb8(255, 255, 255),
      backgroundColor: img.ColorRgb8(0, 0, 0));

  print('Processing ${image.frames.length} frames...');

  // Loop through all frames and annotate them in place
  for (int frameIndex = 0; frameIndex < image.frames.length; frameIndex++) {
    // Get the frame
    final frame = image.frames[frameIndex];

    // Get detections for this frame (if available)
    if (frameIndex < allDetections.length) {
      final frameDetections = allDetections[frameIndex];

      // Annotate frame in place
      boxAnnotator.annotate(frame, frameDetections);
      labelAnnotator.annotate(frame, frameDetections);

      print(
          'Annotated frame $frameIndex with ${frameDetections.length} detections');
    }
  }

  // Save the annotated GIF
  final annotatedGifBytes = img.encodeGif(image);
  await File('assets/people-walking-annotated.gif')
      .writeAsBytes(annotatedGifBytes);

  print('Saved annotated GIF with ${image.frames.length} frames');
}
