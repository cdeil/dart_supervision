/// Simple example of using supervision assets library
import 'package:dart_supervision/dart_supervision.dart';

Future<void> main() async {
  // Download a video asset for computer vision work
  final videoPath = await downloadAssets(VideoAssets.vehicles);
  print('Downloaded video: $videoPath');

  // List all available assets
  print('Available videos: ${VideoAssets.allFilenames}');
}
