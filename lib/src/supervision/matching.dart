import '../numpy/ndarray.dart';
import '../scipy/optimize/linear_assignment.dart';

/// Hungarian algorithm for solving the assignment problem.
///
/// This is used in object tracking to match detections with existing tracks
/// based on cost matrices (typically derived from IoU distances).
class HungarianMatcher {
  /// Solves the linear assignment problem using the Hungarian algorithm.
  ///
  /// Returns matches, unmatched rows, and unmatched columns.
  static (List<List<int>>, List<int>, List<int>) linearAssignment(
    NDArray costMatrix,
    double threshold,
  ) {
    if (costMatrix.ndim != 2) {
      throw ArgumentError('Cost matrix must be 2D');
    }

    final rows = costMatrix.shape[0];
    final cols = costMatrix.shape[1];

    if (rows == 0 || cols == 0) {
      return ([], List.generate(rows, (i) => i), List.generate(cols, (i) => i));
    }

    // Create a cost matrix copy for thresholding
    final workingCost = costMatrix.copy();

    // Set costs above threshold to a high value
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        if (workingCost[[i, j]] > threshold) {
          workingCost[[i, j]] = threshold + 1e-4;
        }
      }
    }

    // Solve the assignment problem using the Hungarian algorithm
    final result = LinearSumAssignment.solve(workingCost);

    // Filter matches by threshold
    final matches = <List<int>>[];
    final usedRows = <bool>[...List.filled(rows, false)];
    final usedCols = <bool>[...List.filled(cols, false)];

    for (int i = 0; i < result.rowIndices.length; i++) {
      final row = result.rowIndices[i];
      final col = result.colIndices[i];
      final cost = costMatrix[[row, col]];

      if (cost <= threshold) {
        matches.add([row, col]);
        usedRows[row] = true;
        usedCols[col] = true;
      }
    }

    // Collect unmatched
    final unmatchedRows = <int>[];
    final unmatchedCols = <int>[];

    for (int i = 0; i < rows; i++) {
      if (!usedRows[i]) unmatchedRows.add(i);
    }

    for (int j = 0; j < cols; j++) {
      if (!usedCols[j]) unmatchedCols.add(j);
    }

    return (matches, unmatchedRows, unmatchedCols);
  }

  /// Fuses detection confidence scores with distance costs.
  static NDArray fuseScore(NDArray distances, List<double> scores) {
    if (distances.shape[1] != scores.length) {
      throw ArgumentError(
        'Number of scores must match number of columns in distance matrix',
      );
    }

    final result = distances.copy();

    for (int i = 0; i < distances.shape[0]; i++) {
      for (int j = 0; j < distances.shape[1]; j++) {
        // Fuse distance with inverse confidence
        final confidence = scores[j];
        final distanceCost = result[[i, j]];
        result[[i, j]] = distanceCost * (2.0 - confidence);
      }
    }

    return result;
  }
}
