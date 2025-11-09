import 'package:dart_supervision/dart_supervision.dart';

void main() {
  print('=== SciPy-like Cholesky Decomposition Demo ===\n');

  // Example 1: Basic Cholesky decomposition
  print('1. Basic Cholesky Decomposition');
  final a = NDArray.fromList([
    [4.0, 2.0, 1.0],
    [2.0, 5.0, 3.0],
    [1.0, 3.0, 6.0],
  ]);

  print('Original matrix A:');
  print(a);

  final chol = LinAlg.choFactor(a, lower: true);
  final l = chol.factorization;

  print('\nCholesky factor L (lower triangular):');
  print(l);

  // Verify L * L^T = A
  final reconstructed = l.dot(l.transpose());
  print('\nReconstructed A = L * L^T:');
  print(reconstructed);

  print(
    '\nVerification: max absolute difference = ${_maxAbsDiff(a, reconstructed)}',
  );

  // Example 2: Solving linear system
  print('\n2. Solving Linear System Ax = b');
  final b = NDArray.fromList([1.0, 2.0, 3.0]);
  print('Right-hand side b:');
  print(b);

  final x = LinAlg.choSolve(chol, b);
  print('\nSolution x:');
  print(x);

  // Verify A * x = b
  final ax = a.dot(x.reshape([3, 1])).flatten();
  print('\nVerification A * x:');
  print(ax);
  print('Max error: ${_maxAbsDiff(b, ax)}');

  // Example 3: Multiple right-hand sides
  print('\n3. Multiple Right-Hand Sides');
  final bMultiple = NDArray.fromList([
    [1.0, 4.0],
    [2.0, 5.0],
    [3.0, 6.0],
  ]);
  print('Multiple RHS matrix B:');
  print(bMultiple);

  final xMultiple = LinAlg.choSolve(chol, bMultiple);
  print('\nSolutions X:');
  print(xMultiple);

  final axMultiple = a.dot(xMultiple);
  print('\nVerification A * X:');
  print(axMultiple);

  // Example 4: Upper triangular decomposition
  print('\n4. Upper Triangular Decomposition');
  final cholUpper = LinAlg.choFactor(a, lower: false);
  final u = cholUpper.factorization;

  print('Cholesky factor U (upper triangular):');
  print(u);

  final reconstructedUpper = u.transpose().dot(u);
  print('\nReconstructed A = U^T * U:');
  print(reconstructedUpper);
  print('Max error: ${_maxAbsDiff(a, reconstructedUpper)}');

  // Example 5: Comparison with standard matrix inversion
  print('\n5. Performance Comparison (conceptual)');
  print('For symmetric positive definite matrices:');
  print('- Cholesky decomposition: O(n³/3) operations');
  print('- Standard inversion: O(n³) operations');
  print('- Cholesky is ~3x faster and more numerically stable!');

  // Example 6: Kalman filter usage
  print('\n6. Kalman Filter Integration');
  print('The Cholesky decomposition is now integrated into the');
  print('KalmanFilter class for computing the Kalman gain:');
  print('K = P * H^T * (H * P * H^T + R)^(-1)');
  print('');
  print('Instead of computing the inverse directly, we solve:');
  print('(H * P * H^T + R) * K^T = (P * H^T)^T');
  print('using cho_solve for better numerical stability.');
}

double _maxAbsDiff(NDArray a, NDArray b) {
  if (a.size != b.size) {
    throw ArgumentError('Arrays must have same size');
  }

  double maxDiff = 0.0;

  if (a.ndim == 1) {
    // Handle 1D arrays
    for (int i = 0; i < a.size; i++) {
      final diff = (a[[i]] - b[[i]]).abs();
      if (diff > maxDiff) maxDiff = diff;
    }
  } else if (a.ndim == 2) {
    // Handle 2D arrays
    for (int i = 0; i < a.shape[0]; i++) {
      for (int j = 0; j < a.shape[1]; j++) {
        final diff = (a[[i, j]] - b[[i, j]]).abs();
        if (diff > maxDiff) maxDiff = diff;
      }
    }
  } else {
    throw ArgumentError('Only 1D and 2D arrays supported');
  }

  return maxDiff;
}
