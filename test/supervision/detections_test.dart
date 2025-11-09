import 'package:test/test.dart';
import 'package:dart_supervision/dart_supervision.dart';

void main() {
  group('Detections', () {
    test('creates empty detections', () {
      final detections = Detections.empty();

      expect(detections.isEmpty, isTrue);
      expect(detections.length, equals(0));
      expect(detections.xyxy.shape, equals([0, 4]));
    });

    test('creates detections with data', () {
      final xyxy = NDArray(
        [2, 4],
        data: [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
      );

      final confidence = NDArray([2], data: [0.9, 0.8]);
      final classId = NDArray([2], data: [1.0, 2.0]);

      final detections = Detections(
        xyxy: xyxy,
        confidence: confidence,
        classId: classId,
      );

      expect(detections.isEmpty, isFalse);
      expect(detections.length, equals(2));
      expect(detections.xyxy.shape, equals([2, 4]));
      expect(detections.confidence!.shape, equals([2]));
      expect(detections.classId!.shape, equals([2]));
    });

    test('validates input dimensions', () {
      final badXyxy = NDArray([2, 3]); // Should be [N, 4]

      expect(() => Detections(xyxy: badXyxy), throwsArgumentError);
    });

    test('validates confidence dimensions', () {
      final xyxy = NDArray([2, 4]);
      final badConfidence = NDArray([3]); // Should match N=2

      expect(
        () => Detections(xyxy: xyxy, confidence: badConfidence),
        throwsArgumentError,
      );
    });

    test('subsets detections by indices', () {
      final xyxy = NDArray(
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
          90.0,
          100.0,
          110.0,
          120.0,
        ],
      );

      final confidence = NDArray([3], data: [0.9, 0.8, 0.7]);

      final detections = Detections(xyxy: xyxy, confidence: confidence);

      final indices = NDArray(
        [2],
        data: [0.0, 2.0],
      ); // First and third detection
      final subset = detections[indices];

      expect(subset.length, equals(2));
      expect(subset.xyxy[[0, 0]], equals(10.0));
      expect(subset.xyxy[[1, 0]], equals(90.0));
      expect(subset.confidence![[0]], equals(0.9));
      expect(subset.confidence![[1]], equals(0.7));
    });

    test('validates subset indices', () {
      final detections = Detections(xyxy: NDArray([2, 4]));
      final badIndices = NDArray([1], data: [5.0]); // Index out of bounds

      expect(() => detections[badIndices], throwsRangeError);
    });

    test('gets anchor coordinates', () {
      final xyxy = NDArray(
        [2, 4],
        data: [
          0.0, 0.0, 10.0, 10.0, // Box 1: center = (5, 5)
          20.0, 30.0, 40.0, 50.0, // Box 2: center = (30, 40)
        ],
      );

      final detections = Detections(xyxy: xyxy);

      final centers = detections.getAnchorsCoordinates(anchor: 'center');
      expect(centers.shape, equals([2, 2]));
      expect(centers[[0, 0]], equals(5.0));
      expect(centers[[0, 1]], equals(5.0));
      expect(centers[[1, 0]], equals(30.0));
      expect(centers[[1, 1]], equals(40.0));

      final topLeft = detections.getAnchorsCoordinates(anchor: 'top_left');
      expect(topLeft[[0, 0]], equals(0.0));
      expect(topLeft[[0, 1]], equals(0.0));

      final bottomCenter = detections.getAnchorsCoordinates(
        anchor: 'bottom_center',
      );
      expect(bottomCenter[[0, 0]], equals(5.0));
      expect(bottomCenter[[0, 1]], equals(10.0));
    });

    test('applies NMS filtering', () {
      final xyxy = NDArray(
        [3, 4],
        data: [
          0.0, 0.0, 10.0, 10.0, // Box 1
          1.0, 1.0, 11.0, 11.0, // Box 2: high overlap with Box 1
          20.0, 20.0, 30.0, 30.0, // Box 3: no overlap
        ],
      );

      final confidence = NDArray([3], data: [0.9, 0.8, 0.7]);

      final detections = Detections(xyxy: xyxy, confidence: confidence);
      final filtered = detections.withNms(threshold: 0.5);

      expect(filtered.length, equals(2)); // Should keep boxes 1 and 3
      expect(
        filtered.confidence![[0]],
        equals(0.9),
      ); // Highest confidence first
    });

    test('handles empty detections in anchor coordinates', () {
      final empty = Detections.empty();
      final anchors = empty.getAnchorsCoordinates();

      expect(anchors.shape, equals([0, 2]));
    });
  });
}
