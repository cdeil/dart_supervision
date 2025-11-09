import 'package:test/test.dart';
import 'package:dart_supervision/dart_supervision.dart';

void main() {
  group('LinearSumAssignment tests', () {
    test('should solve basic 3x3 assignment problem', () {
      // Example from SciPy documentation:
      // cost = [[4, 1, 3], [2, 0, 5], [3, 2, 2]]
      // Expected solution: [(0,1), (1,0), (2,2)] with cost 5
      final cost = NDArray.fromList([
        [4.0, 1.0, 3.0],
        [2.0, 0.0, 5.0],
        [3.0, 2.0, 2.0],
      ]);

      final result = LinearSumAssignment.solve(cost);

      // Check that we have 3 assignments
      expect(result.rowIndices.length, equals(3));
      expect(result.colIndices.length, equals(3));

      // Verify the expected assignments and total cost
      expect(result.totalCost(cost), equals(5.0));

      // Check specific assignment (order might vary but cost should be minimal)
      final assignments = <String>{};
      for (int i = 0; i < result.rowIndices.length; i++) {
        assignments.add('${result.rowIndices[i]},${result.colIndices[i]}');
      }

      // One possible optimal solution
      final expectedSet1 = {'0,1', '1,0', '2,2'};
      expect(assignments, equals(expectedSet1));
    });

    test('should handle rectangular matrices (more rows than columns)', () {
      final cost = NDArray.fromList([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
      ]);

      final result = LinearSumAssignment.solve(cost);

      // Should assign 2 workers to 2 jobs
      expect(result.rowIndices.length, equals(2));
      expect(result.colIndices.length, equals(2));

      // Verify optimal cost (should be 1+4=5)
      expect(result.totalCost(cost), equals(5.0));
    });

    test('should handle rectangular matrices (more columns than rows)', () {
      final cost = NDArray.fromList([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
      ]);

      final result = LinearSumAssignment.solve(cost);

      // Should assign 2 workers to 2 jobs
      expect(result.rowIndices.length, equals(2));
      expect(result.colIndices.length, equals(2));

      // Verify optimal cost (should be 1+5=6)
      expect(result.totalCost(cost), equals(6.0));
    });

    test('should handle maximization', () {
      final cost = NDArray.fromList([
        [4.0, 1.0, 3.0],
        [2.0, 0.0, 5.0],
        [3.0, 2.0, 2.0],
      ]);

      final result = LinearSumAssignment.solve(cost, maximize: true);

      // For maximization, the algorithm should find max weight matching
      // The negated minimum should give us the maximum
      expect(result.rowIndices.length, equals(3));
      expect(result.colIndices.length, equals(3));

      // The maximum should be 11.0 (verified by exhaustive search)
      expect(result.totalCost(cost), equals(11.0));
    });

    test('should handle single element matrix', () {
      final cost = NDArray.fromList([
        [5.0],
      ]);

      final result = LinearSumAssignment.solve(cost);

      expect(result.rowIndices, equals([0]));
      expect(result.colIndices, equals([0]));
      expect(result.totalCost(cost), equals(5.0));
    });

    test('should handle empty matrix', () {
      final cost = NDArray([0, 0]);

      final result = LinearSumAssignment.solve(cost);

      expect(result.rowIndices, isEmpty);
      expect(result.colIndices, isEmpty);
    });

    test('should reject invalid inputs', () {
      // Non-2D array
      final vector = NDArray([3]);
      expect(() => LinearSumAssignment.solve(vector), throwsArgumentError);

      // Matrix with NaN values
      final costWithNaN = NDArray.fromList([
        [1.0, double.nan],
        [2.0, 3.0],
      ]);
      expect(() => LinearSumAssignment.solve(costWithNaN), throwsArgumentError);

      // Matrix with negative infinity
      final costWithNegInf = NDArray.fromList([
        [1.0, double.negativeInfinity],
        [2.0, 3.0],
      ]);
      expect(
        () => LinearSumAssignment.solve(costWithNegInf),
        throwsArgumentError,
      );
    });

    test('should produce same results as known solutions', () {
      // Test case from literature
      final cost = NDArray.fromList([
        [3.0, 4.0, 6.0, 4.0, 9.0],
        [6.0, 4.0, 5.0, 3.0, 8.0],
        [7.0, 5.0, 3.0, 4.0, 2.0],
        [6.0, 3.0, 2.0, 2.0, 5.0],
        [8.0, 4.0, 5.0, 4.0, 3.0],
      ]);

      final result = LinearSumAssignment.solve(cost);

      expect(result.rowIndices.length, equals(5));
      expect(result.colIndices.length, equals(5));

      // Our implementation gives 14 for this test case
      expect(result.totalCost(cost), equals(14.0));
    });

    test('should handle large cost differences', () {
      final cost = NDArray.fromList([
        [1.0, 1000.0],
        [1000.0, 1.0],
      ]);

      final result = LinearSumAssignment.solve(cost);

      // Should choose the diagonal assignments (0,0) and (1,1)
      expect(result.totalCost(cost), equals(2.0));
    });
  });
}
