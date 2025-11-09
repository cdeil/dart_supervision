import 'package:test/test.dart';
import 'package:dart_supervision/dart_supervision.dart';
import 'dart:io';

void main() {
  group('DetectionUtils', () {
    test('should load detection data correctly', () async {
      // Check if the detection file exists
      final file = File('assets/people-walking-detections.json');
      if (!await file.exists()) {
        // Skip test if detection file doesn't exist
        print(
            'Skipping test - detection file not found. Run generate_detections.py first.');
        return;
      }

      // Load detections from first frame
      final detections = await DetectionUtils.loadFromJson(
        'assets/people-walking-detections.json',
        0,
      );

      expect(detections.length, greaterThan(0));
      expect(
          detections.xyxy.shape[1], equals(4)); // 4 coordinates per detection
      expect(detections.confidence, isNotNull);
      expect(detections.trackerId, isNotNull);

      // Get video info to check coordinate bounds (coordinates should be in pixel space)
      final videoInfo = await DetectionUtils.getVideoInfo(
        'assets/people-walking-detections.json',
      );
      final videoData = videoInfo['video_info'] as Map;
      final imageWidth = videoData['width'] as int;
      final imageHeight = videoData['height'] as int;

      // Verify pixel coordinates (should be between 0 and image dimensions)
      for (int i = 0; i < detections.length; i++) {
        final x1 = detections.xyxy[[i, 0]];
        final y1 = detections.xyxy[[i, 1]];
        final x2 = detections.xyxy[[i, 2]];
        final y2 = detections.xyxy[[i, 3]];

        expect(x1, inInclusiveRange(0.0, imageWidth.toDouble()),
            reason: 'X1 coordinate should be within image width');
        expect(x2, inInclusiveRange(0.0, imageWidth.toDouble()),
            reason: 'X2 coordinate should be within image width');
        expect(y1, inInclusiveRange(0.0, imageHeight.toDouble()),
            reason: 'Y1 coordinate should be within image height');
        expect(y2, inInclusiveRange(0.0, imageHeight.toDouble()),
            reason: 'Y2 coordinate should be within image height');
      }

      print('Loaded ${detections.length} detections from frame 0');
    });

    test('should load video info correctly', () async {
      final file = File('assets/people-walking-detections.json');
      if (!await file.exists()) {
        print('Skipping test - detection file not found');
        return;
      }

      final videoInfo = await DetectionUtils.getVideoInfo(
        'assets/people-walking-detections.json',
      );

      expect(videoInfo, containsPair('video_info', isA<Map>()));
      expect(videoInfo, containsPair('model_info', isA<Map>()));
      expect(videoInfo, containsPair('total_frames', isA<int>()));

      final videoData = videoInfo['video_info'] as Map;
      expect(videoData, containsPair('width', isA<num>()));
      expect(videoData, containsPair('height', isA<num>()));
      expect(videoData, containsPair('fps', isA<num>()));

      print(
          'Video info: ${videoData['width']}x${videoData['height']} @ ${videoData['fps']} fps');
    });

    test('should normalize and denormalize coordinates correctly', () {
      // Create test detections with pixel coordinates
      final pixelDetections = Detections(
        xyxy: NDArray([
          2,
          4
        ], data: [
          100.0, 200.0, 300.0, 400.0, // Detection 1: (100,200) to (300,400)
          50.0, 100.0, 150.0, 200.0, // Detection 2: (50,100) to (150,200)
        ]),
      );

      const imageWidth = 1920;
      const imageHeight = 1080;

      // Normalize coordinates
      final normalized = DetectionUtils.normalizeCoordinates(
        pixelDetections,
        imageWidth,
        imageHeight,
      );

      // Check that coordinates are normalized
      expect(normalized.xyxy[[0, 0]], closeTo(100.0 / imageWidth, 0.001));
      expect(normalized.xyxy[[0, 1]], closeTo(200.0 / imageHeight, 0.001));
      expect(normalized.xyxy[[1, 2]], closeTo(150.0 / imageWidth, 0.001));
      expect(normalized.xyxy[[1, 3]], closeTo(200.0 / imageHeight, 0.001));

      // Denormalize back to pixels
      final denormalized = DetectionUtils.denormalizeCoordinates(
        normalized,
        imageWidth,
        imageHeight,
      );

      // Should match original pixel coordinates
      expect(denormalized.xyxy[[0, 0]], closeTo(100.0, 0.1));
      expect(denormalized.xyxy[[0, 1]], closeTo(200.0, 0.1));
      expect(denormalized.xyxy[[1, 2]], closeTo(150.0, 0.1));
      expect(denormalized.xyxy[[1, 3]], closeTo(200.0, 0.1));
    });
  });
}
