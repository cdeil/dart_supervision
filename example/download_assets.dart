/// Simple example of using supervision assets library
import 'package:dart_supervision/dart_supervision.dart' as sv;

Future<void> main() async {
  // Download a video asset for computer vision work
  final videoPath = await sv.downloadAssets(sv.VideoAssets.vehicles);
  print('Downloaded video: $videoPath');

  // List all available assets
  print('Available videos: ${sv.VideoAssets.allFilenames}');
}
