import 'package:test/test.dart';
import 'package:dart_supervision/dart_supervision.dart';

void main() {
  group('ByteTracker', () {
    test('creates tracker with default parameters', () {
      final tracker = ByteTracker();

      expect(tracker.trackActivationThreshold, equals(0.25));
      expect(tracker.minimumMatchingThreshold, equals(0.8));
      expect(tracker.frameRate, equals(30));
      expect(tracker.lostTrackBuffer, equals(30));
    });

    test('handles empty detections', () {
      final tracker = ByteTracker();
      final emptyDetections = Detections.empty();

      final result = tracker.updateWithDetections(emptyDetections);

      expect(result.isEmpty, isTrue);
    });

    test('creates new tracks from high confidence detections', () {
      final tracker = ByteTracker(
        trackActivationThreshold: 0.5,
      ); // Lower threshold

      final detections = Detections(
        xyxy: NDArray(
          [2, 4],
          data: [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        ),
        confidence: NDArray([2], data: [0.9, 0.8]),
      );

      final result = tracker.updateWithDetections(detections);

      expect(result.length, greaterThan(0));
      expect(result.trackerId, isNotNull);
    });

    test('ignores low confidence detections', () {
      final tracker = ByteTracker();

      final lowConfDetections = Detections(
        xyxy: NDArray([1, 4], data: [10.0, 20.0, 30.0, 40.0]),
        confidence: NDArray([1], data: [0.1]), // Below threshold
      );

      final result = tracker.updateWithDetections(lowConfDetections);

      expect(result.isEmpty, isTrue);
    });

    test('tracks objects across frames', () {
      final tracker = ByteTracker(
        trackActivationThreshold: 0.5, // Lower threshold
        minimumMatchingThreshold: 0.3, // More lenient matching
      );

      // Frame 1: Initial detections
      final frame1 = Detections(
        xyxy: NDArray([1, 4], data: [10.0, 20.0, 30.0, 40.0]),
        confidence: NDArray([1], data: [0.9]),
      );

      final result1 = tracker.updateWithDetections(frame1);
      expect(result1.length, equals(1));

      // Frame 2: Object moved slightly
      final frame2 = Detections(
        xyxy: NDArray([1, 4], data: [11.0, 21.0, 31.0, 41.0]),
        confidence: NDArray([1], data: [0.8]),
      );

      final result2 = tracker.updateWithDetections(frame2);
      expect(result2.length, equals(1));

      // Track ID should be maintained
      if (result1.trackerId != null && result2.trackerId != null) {
        expect(result2.trackerId![[0]], equals(result1.trackerId![[0]]));
      }
    });

    test('resets tracker state', () {
      final tracker = ByteTracker(trackActivationThreshold: 0.5);

      // Add some detections to create tracks
      final detections = Detections(
        xyxy: NDArray([1, 4], data: [10.0, 20.0, 30.0, 40.0]),
        confidence: NDArray([1], data: [0.9]),
      );

      tracker.updateWithDetections(detections);

      // Reset and verify state is clean
      tracker.reset();

      final result2 = tracker.updateWithDetections(detections);
      if (result2.trackerId != null && result2.length > 0) {
        expect(
          result2.trackerId![[0]],
          equals(1.0),
        ); // Should start from ID 1 again
      }
    });

    test('handles multiple objects', () {
      final tracker = ByteTracker(trackActivationThreshold: 0.5);

      final detections = Detections(
        xyxy: NDArray(
          [3, 4],
          data: [
            10.0,
            20.0,
            30.0,
            40.0,
            50.0,
            60.0,
            70.0,
            80.0,
            100.0,
            110.0,
            120.0,
            130.0,
          ],
        ),
        confidence: NDArray([3], data: [0.9, 0.8, 0.7]),
      );

      final result = tracker.updateWithDetections(detections);

      expect(result.length, equals(3));
      expect(result.trackerId, isNotNull);

      // All tracks should have unique IDs
      final trackIds = <double>{};
      for (int i = 0; i < result.length; i++) {
        trackIds.add(result.trackerId![[i]]);
      }
      expect(trackIds.length, equals(3));
    });

    test('string representation', () {
      final tracker = ByteTracker();
      final str = tracker.toString();

      expect(str, contains('ByteTracker'));
      expect(str, contains('tracked'));
      expect(str, contains('lost'));
      expect(str, contains('frame'));
    });
  });

  group('Integration Tests', () {
    test('tracks multiple objects through multiple frames', () {
      final tracker = ByteTracker(
        trackActivationThreshold: 0.5,
        minimumMatchingThreshold: 0.3,
      );
      final results = <Detections>[];

      // Simulate 5 frames with 2 objects moving
      for (int frame = 0; frame < 5; frame++) {
        final offset = frame * 1.0; // Small movement
        final detections = Detections(
          xyxy: NDArray(
            [2, 4],
            data: [
              10.0 + offset,
              20.0 + offset,
              30.0 + offset,
              40.0 + offset,
              100.0 - offset,
              120.0 - offset,
              130.0 - offset,
              140.0 - offset,
            ],
          ),
          confidence: NDArray([2], data: [0.9, 0.8]),
        );

        final result = tracker.updateWithDetections(detections);
        results.add(result);
      }

      // All frames should have tracked objects
      for (int i = 0; i < results.length; i++) {
        expect(
          results[i].length,
          greaterThan(0),
          reason: 'Frame $i should have detections',
        );
      }

      // First frame should establish track IDs
      expect(results[0].trackerId, isNotNull);
      if (results[0].length >= 2) {
        final firstFrameIds = <double>{
          results[0].trackerId![[0]],
          if (results[0].length > 1) results[0].trackerId![[1]],
        };

        // Later frames should maintain some of these IDs
        bool hasConsistentTracking = false;
        for (int i = 1; i < results.length; i++) {
          if (results[i].trackerId != null && results[i].length > 0) {
            for (int j = 0; j < results[i].length; j++) {
              if (firstFrameIds.contains(results[i].trackerId![[j]])) {
                hasConsistentTracking = true;
                break;
              }
            }
          }
        }
        expect(
          hasConsistentTracking,
          isTrue,
          reason: 'Should maintain at least some track IDs across frames',
        );
      }
    });

    test('demonstrates supervision-like workflow', () {
      final tracker = ByteTracker(trackActivationThreshold: 0.5);

      // Simulate video processing workflow
      for (int frame = 0; frame < 10; frame++) {
        // Create mock detections (like from YOLO)
        final xyxy = NDArray(
          [2, 4],
          data: [
            10.0 + frame, 20.0 + frame, 50.0 + frame, 60.0 + frame,
            100.0, 100.0, 150.0, 150.0, // Stationary object
          ],
        );

        final confidence = NDArray([2], data: [0.8, 0.9]);
        final classId = NDArray([2], data: [0.0, 1.0]); // person, car

        final detections = Detections(
          xyxy: xyxy,
          confidence: confidence,
          classId: classId,
        );

        // Apply NMS (like supervision does)
        final filteredDetections = detections.withNms(threshold: 0.5);

        // Update tracker (like supervision ByteTrack)
        final trackedDetections = tracker.updateWithDetections(
          filteredDetections,
        );

        // Get anchor points for visualization (like supervision)
        if (trackedDetections.length > 0) {
          final anchors = trackedDetections.getAnchorsCoordinates(
            anchor: 'bottom_center',
          );
          expect(anchors.shape[0], equals(trackedDetections.length));
          expect(anchors.shape[1], equals(2));
        }

        // Verify tracker IDs are assigned
        if (trackedDetections.length > 0) {
          expect(trackedDetections.trackerId, isNotNull);
        }
      }
    });
  });
}
