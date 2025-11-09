import 'package:test/test.dart';
import 'package:dart_supervision/dart_supervision.dart';

void main() {
  group('IoU Calculation', () {
    test('calculates IoU correctly for overlapping boxes', () {
      final box1 = NDArray([4], data: [0.0, 0.0, 10.0, 10.0]); // 100 area
      final box2 = NDArray(
        [4],
        data: [5.0, 5.0, 15.0, 15.0],
      ); // 100 area, 25 overlap

      final iou = IoUCalculator.iou(box1, box2);
      expect(iou, closeTo(25.0 / 175.0, 0.001)); // intersection / union
    });

    test('returns 0 for non-overlapping boxes', () {
      final box1 = NDArray([4], data: [0.0, 0.0, 5.0, 5.0]);
      final box2 = NDArray([4], data: [10.0, 10.0, 15.0, 15.0]);

      final iou = IoUCalculator.iou(box1, box2);
      expect(iou, equals(0.0));
    });

    test('returns 1 for identical boxes', () {
      final box1 = NDArray([4], data: [0.0, 0.0, 10.0, 10.0]);
      final box2 = NDArray([4], data: [0.0, 0.0, 10.0, 10.0]);

      final iou = IoUCalculator.iou(box1, box2);
      expect(iou, equals(1.0));
    });

    test('computes IoU matrix between box sets', () {
      final boxes1 = NDArray(
        [2, 4],
        data: [
          0.0, 0.0, 10.0, 10.0, // Box 1
          20.0, 20.0, 30.0, 30.0, // Box 2
        ],
      );

      final boxes2 = NDArray(
        [2, 4],
        data: [
          5.0, 5.0, 15.0, 15.0, // Overlaps with Box 1
          25.0, 25.0, 35.0, 35.0, // Overlaps with Box 2
        ],
      );

      final iouMat = IoUCalculator.iouBatch(boxes1, boxes2);

      expect(iouMat.shape, equals([2, 2]));
      expect(iouMat[[0, 0]], greaterThan(0)); // Box 1 overlaps with first box2
      expect(iouMat[[1, 1]], greaterThan(0)); // Box 2 overlaps with second box2
      expect(
        iouMat[[0, 1]],
        equals(0),
      ); // Box 1 doesn't overlap with second box2
      expect(
        iouMat[[1, 0]],
        equals(0),
      ); // Box 2 doesn't overlap with first box2
    });

    test('computes IoU distance matrix', () {
      final boxes1 = NDArray([1, 4], data: [0.0, 0.0, 10.0, 10.0]);
      final boxes2 = NDArray([1, 4], data: [0.0, 0.0, 10.0, 10.0]);

      final distances = IoUCalculator.iouDistance(boxes1, boxes2);
      expect(distances.shape, equals([1, 1]));
      expect(distances[[0, 0]], equals(0.0)); // distance = 1 - IoU = 1 - 1 = 0
    });
  });
}
