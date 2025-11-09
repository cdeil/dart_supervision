import 'package:test/test.dart';
import 'package:dart_supervision/dart_supervision.dart';

void main() {
  group('Hungarian Assignment', () {
    test('finds optimal assignment', () {
      final costMatrix = NDArray(
        [3, 3],
        data: [0.1, 0.9, 0.8, 0.7, 0.2, 0.6, 0.5, 0.4, 0.3],
      );

      final result = HungarianMatcher.linearAssignment(costMatrix, 0.5);
      final matches = result.$1;

      expect(matches.length, greaterThan(0));
      expect(matches.length, lessThanOrEqualTo(3));

      // Verify no duplicate assignments
      final usedRows = matches.map((m) => m[0]).toSet();
      final usedCols = matches.map((m) => m[1]).toSet();
      expect(usedRows.length, equals(matches.length));
      expect(usedCols.length, equals(matches.length));
    });

    test('fuses confidence scores with distances', () {
      final distances = NDArray([2, 2], data: [0.5, 0.8, 0.3, 0.9]);
      final scores = [0.9, 0.7];

      final fused = HungarianMatcher.fuseScore(distances, scores);
      expect(fused.shape, equals([2, 2]));
      // Fused cost should be distance * (2 - confidence)
      expect(fused[[0, 0]], closeTo(0.5 * (2.0 - 0.9), 0.001));
      expect(fused[[0, 1]], closeTo(0.8 * (2.0 - 0.7), 0.001));
    });
  });
}
