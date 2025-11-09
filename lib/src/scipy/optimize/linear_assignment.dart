import '../../numpy/ndarray.dart';

/// Result of linear sum assignment algorithm.
class LinearAssignmentResult {
  /// Row indices of the optimal assignment.
  final List<int> rowIndices;

  /// Column indices of the optimal assignment.
  final List<int> colIndices;

  const LinearAssignmentResult({
    required this.rowIndices,
    required this.colIndices,
  });

  /// Returns the total cost of the assignment.
  double totalCost(NDArray costMatrix) {
    double cost = 0.0;
    for (int i = 0; i < rowIndices.length; i++) {
      cost += costMatrix[[rowIndices[i], colIndices[i]]];
    }
    return cost;
  }
}

/// Solves the linear sum assignment problem using the Hungarian algorithm.
///
/// This implementation is based on the Jonker-Volgenant algorithm used by SciPy,
/// which implements the shortest augmenting path approach.
class LinearSumAssignment {
  /// Solve the linear sum assignment problem.
  ///
  /// Parameters:
  /// - [costMatrix]: The cost matrix of the bipartite graph (2D NDArray)
  /// - [maximize]: If true, calculates a maximum weight matching (default: false)
  ///
  /// Returns a [LinearAssignmentResult] containing the optimal assignment.
  ///
  /// The linear sum assignment problem is also known as minimum weight matching
  /// in bipartite graphs. A problem instance is described by a matrix C, where
  /// each C[i,j] is the cost of matching vertex i of the first partite set
  /// (a 'worker') and vertex j of the second set (a 'job'). The goal is to find
  /// a complete assignment of workers to jobs of minimal cost.
  ///
  /// This function can also solve a generalization of the classic assignment
  /// problem where the cost matrix is rectangular. If it has more rows than
  /// columns, then not every row needs to be assigned to a column, and vice versa.
  static LinearAssignmentResult solve(
    NDArray costMatrix, {
    bool maximize = false,
  }) {
    if (costMatrix.ndim != 2) {
      throw ArgumentError('Cost matrix must be 2D, got ${costMatrix.ndim}D');
    }

    int nr = costMatrix.shape[0]; // number of rows (workers)
    int nc = costMatrix.shape[1]; // number of columns (jobs)

    // Handle trivial cases
    if (nr == 0 || nc == 0) {
      return LinearAssignmentResult(rowIndices: [], colIndices: []);
    }

    // Create working cost matrix
    NDArray workingCost = costMatrix.copy();
    bool transpose = nc < nr;

    // If tall matrix (more rows than columns), transpose it
    if (transpose) {
      workingCost = _transpose(workingCost);
      int temp = nr;
      nr = nc;
      nc = temp;
    }

    // Negate for maximization
    if (maximize) {
      for (int i = 0; i < nr; i++) {
        for (int j = 0; j < nc; j++) {
          workingCost[[i, j]] = -workingCost[[i, j]];
        }
      }
    }

    // Check for invalid entries (NaN or -infinity)
    for (int i = 0; i < nr; i++) {
      for (int j = 0; j < nc; j++) {
        double val = workingCost[[i, j]];
        if (val.isNaN || val.isInfinite && val.isNegative) {
          throw ArgumentError('Cost matrix contains invalid numeric entries');
        }
      }
    }

    // Solve the assignment problem
    var result = _solveAssignment(workingCost, nr, nc);

    // Handle transposition in result
    if (transpose) {
      // When we transposed, we need to swap row and column indices
      // and sort by the original row indices
      var pairs = <MapEntry<int, int>>[];
      for (int i = 0; i < result.colIndices.length; i++) {
        pairs.add(MapEntry(result.colIndices[i], result.rowIndices[i]));
      }
      pairs.sort((a, b) => a.key.compareTo(b.key));

      return LinearAssignmentResult(
        rowIndices: pairs.map((p) => p.key).toList(),
        colIndices: pairs.map((p) => p.value).toList(),
      );
    }

    return result;
  }

  /// Transpose a 2D matrix.
  static NDArray _transpose(NDArray matrix) {
    int rows = matrix.shape[0];
    int cols = matrix.shape[1];
    NDArray result = NDArray([cols, rows]);

    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        result[[j, i]] = matrix[[i, j]];
      }
    }

    return result;
  }

  /// Core assignment algorithm using the Jonker-Volgenant shortest augmenting path method.
  static LinearAssignmentResult _solveAssignment(NDArray cost, int nr, int nc) {
    // Initialize dual variables and assignment arrays
    var u = List<double>.filled(nr, 0.0); // dual variables for rows
    var v = List<double>.filled(nc, 0.0); // dual variables for columns
    var col4row = List<int>.filled(nr, -1); // column assigned to each row
    var row4col = List<int>.filled(nc, -1); // row assigned to each column

    // Temporary arrays for shortest path computation
    var shortestPathCosts = List<double>.filled(nc, 0.0);
    var path = List<int>.filled(nc, -1);
    var SR = List<bool>.filled(nr, false); // scanned rows
    var SC = List<bool>.filled(nc, false); // scanned columns
    var remaining = List<int>.filled(nc, 0); // remaining unscanned columns

    // Iteratively build the solution
    for (int curRow = 0; curRow < nr; curRow++) {
      var result = _augmentingPath(
        cost,
        nc,
        u,
        v,
        path,
        row4col,
        shortestPathCosts,
        curRow,
        SR,
        SC,
        remaining,
      );

      double minVal = result.$1;
      int sink = result.$2;

      if (sink < 0) {
        throw StateError('Cost matrix is infeasible');
      }

      // Update dual variables
      u[curRow] += minVal;
      for (int i = 0; i < nr; i++) {
        if (SR[i] && i != curRow) {
          u[i] += minVal - shortestPathCosts[col4row[i]];
        }
      }

      for (int j = 0; j < nc; j++) {
        if (SC[j]) {
          v[j] -= minVal - shortestPathCosts[j];
        }
      }

      // Augment the solution by following the path
      int j = sink;
      while (true) {
        int i = path[j];
        row4col[j] = i;
        int temp = col4row[i];
        col4row[i] = j;
        j = temp;
        if (i == curRow) break;
      }
    }

    // Build result arrays
    var rowIndices = <int>[];
    var colIndices = <int>[];

    for (int i = 0; i < nr; i++) {
      if (col4row[i] != -1) {
        rowIndices.add(i);
        colIndices.add(col4row[i]);
      }
    }

    return LinearAssignmentResult(
      rowIndices: rowIndices,
      colIndices: colIndices,
    );
  }

  /// Find shortest augmenting path using Dijkstra-like algorithm.
  static (double, int) _augmentingPath(
    NDArray cost,
    int nc,
    List<double> u,
    List<double> v,
    List<int> path,
    List<int> row4col,
    List<double> shortestPathCosts,
    int startRow,
    List<bool> SR,
    List<bool> SC,
    List<int> remaining,
  ) {
    double minVal = 0.0;

    // Initialize remaining columns (in reverse order for consistency with SciPy)
    int numRemaining = nc;
    for (int it = 0; it < nc; it++) {
      remaining[it] = nc - it - 1;
    }

    // Reset arrays
    for (int i = 0; i < SR.length; i++) SR[i] = false;
    for (int i = 0; i < SC.length; i++) SC[i] = false;
    for (int i = 0; i < shortestPathCosts.length; i++) {
      shortestPathCosts[i] = double.infinity;
    }

    int sink = -1;
    int currentRow = startRow;

    while (sink == -1) {
      int index = -1;
      double lowest = double.infinity;
      SR[currentRow] = true;

      // Update shortest path costs for all remaining columns
      for (int it = 0; it < numRemaining; it++) {
        int j = remaining[it];

        double reducedCost =
            minVal + cost[[currentRow, j]] - u[currentRow] - v[j];
        if (reducedCost < shortestPathCosts[j]) {
          path[j] = currentRow;
          shortestPathCosts[j] = reducedCost;
        }

        // Find the column with minimum cost
        // Prefer unassigned columns when costs are equal
        if (shortestPathCosts[j] < lowest ||
            (shortestPathCosts[j] == lowest && row4col[j] == -1)) {
          lowest = shortestPathCosts[j];
          index = it;
        }
      }

      minVal = lowest;
      if (minVal == double.infinity) {
        return (-1.0, -1); // infeasible
      }

      int j = remaining[index];
      if (row4col[j] == -1) {
        sink = j; // found an unassigned column
      } else {
        currentRow = row4col[j]; // continue with the assigned row
      }

      SC[j] = true;
      // Remove j from remaining by swapping with last element
      remaining[index] = remaining[--numRemaining];
    }

    return (minVal, sink);
  }
}
