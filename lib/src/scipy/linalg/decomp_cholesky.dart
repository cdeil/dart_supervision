import 'dart:math' as math;
import '../../numpy/ndarray.dart';

/// Exception thrown when linear algebra operations fail.
class LinAlgError extends Error {
  final String message;
  LinAlgError(this.message);

  @override
  String toString() => 'LinAlgError: $message';
}

/// Cholesky decomposition result.
class CholeskyFactorization {
  final NDArray factorization;
  final bool lower;

  const CholeskyFactorization(this.factorization, this.lower);
}

/// SciPy-compatible linear algebra functions for Cholesky decomposition.
///
/// This class provides Dart implementations of scipy.linalg.cho_factor
/// and scipy.linalg.cho_solve functions used for efficient solving of
/// linear systems with symmetric positive-definite matrices.
class LinAlg {
  /// Compute the Cholesky decomposition of a matrix, to use in cho_solve.
  ///
  /// Returns a matrix containing the Cholesky decomposition,
  /// A = L L* or A = U* U of a Hermitian positive-definite matrix [a].
  /// The return value can be directly used as the first parameter to cho_solve.
  ///
  /// Parameters:
  /// - [a]: Matrix to be decomposed (must be square and positive-definite)
  /// - [lower]: Whether to compute lower or upper triangular factorization
  ///
  /// Returns:
  /// - [CholeskyFactorization] containing the factor matrix and lower flag
  ///
  /// Throws:
  /// - [LinAlgError] if decomposition fails (matrix not positive-definite)
  /// - [ArgumentError] if matrix is not square
  ///
  /// Example:
  /// ```dart
  /// final a = NDArray.fromList([[4.0, 2.0], [2.0, 3.0]]);
  /// final chol = LinAlg.choFactor(a);
  /// ```
  static CholeskyFactorization choFactor(NDArray a, {bool lower = false}) {
    if (a.ndim != 2) {
      throw ArgumentError('Input array must be 2D, got ${a.ndim}D');
    }

    if (a.shape[0] != a.shape[1]) {
      throw ArgumentError('Input array must be square, got shape ${a.shape}');
    }

    final n = a.shape[0];
    if (n == 0) {
      return CholeskyFactorization(NDArray([0, 0]), lower);
    }

    // Create a copy to avoid modifying the original
    final matrix = a.copy();

    if (lower) {
      _choleskyLower(matrix);
    } else {
      _choleskyUpper(matrix);
    }

    return CholeskyFactorization(matrix, lower);
  }

  /// Solve the linear equations A x = b, given the Cholesky factorization of A.
  ///
  /// Parameters:
  /// - [factorization]: Cholesky factorization from cho_factor
  /// - [b]: Right-hand side vector or matrix
  ///
  /// Returns:
  /// - [NDArray] solution x to the system A x = b
  ///
  /// Throws:
  /// - [ArgumentError] if dimensions are incompatible
  ///
  /// Example:
  /// ```dart
  /// final a = NDArray.fromList([[4.0, 2.0], [2.0, 3.0]]);
  /// final b = NDArray.fromList([2.0, 1.0]);
  /// final chol = LinAlg.choFactor(a);
  /// final x = LinAlg.choSolve(chol, b);
  /// ```
  static NDArray choSolve(CholeskyFactorization factorization, NDArray b) {
    final c = factorization.factorization;
    final lower = factorization.lower;

    if (c.ndim != 2 || c.shape[0] != c.shape[1]) {
      throw ArgumentError('The factored matrix c is not square');
    }

    final n = c.shape[0];

    // Handle vector input
    final isVector = b.ndim == 1;
    NDArray b2d;
    if (isVector) {
      if (b.shape[0] != n) {
        throw ArgumentError(
          'Incompatible dimensions: ${c.shape} and ${b.shape}',
        );
      }
      b2d = b.reshape([n, 1]);
    } else {
      if (b.shape[0] != n) {
        throw ArgumentError(
          'Incompatible dimensions: ${c.shape} and ${b.shape}',
        );
      }
      b2d = b.copy();
    }

    final nrhs = b2d.shape[1];
    final x = b2d.copy();

    if (n == 0) {
      return isVector ? x.flatten() : x;
    }

    if (lower) {
      // Solve L y = b by forward substitution
      for (int col = 0; col < nrhs; col++) {
        for (int i = 0; i < n; i++) {
          double sum = x[[i, col]];
          for (int j = 0; j < i; j++) {
            sum -= c[[i, j]] * x[[j, col]];
          }
          x[[i, col]] = sum / c[[i, i]];
        }
      }

      // Solve L^T x = y by backward substitution
      for (int col = 0; col < nrhs; col++) {
        for (int i = n - 1; i >= 0; i--) {
          double sum = x[[i, col]];
          for (int j = i + 1; j < n; j++) {
            sum -= c[[j, i]] * x[[j, col]];
          }
          x[[i, col]] = sum / c[[i, i]];
        }
      }
    } else {
      // Solve U^T y = b by forward substitution
      for (int col = 0; col < nrhs; col++) {
        for (int i = 0; i < n; i++) {
          double sum = x[[i, col]];
          for (int j = 0; j < i; j++) {
            sum -= c[[j, i]] * x[[j, col]];
          }
          x[[i, col]] = sum / c[[i, i]];
        }
      }

      // Solve U x = y by backward substitution
      for (int col = 0; col < nrhs; col++) {
        for (int i = n - 1; i >= 0; i--) {
          double sum = x[[i, col]];
          for (int j = i + 1; j < n; j++) {
            sum -= c[[i, j]] * x[[j, col]];
          }
          x[[i, col]] = sum / c[[i, i]];
        }
      }
    }

    return isVector ? x.flatten() : x;
  }

  /// Performs in-place Cholesky decomposition for lower triangular matrix.
  ///
  /// Computes L such that A = L L^T where L is lower triangular.
  static void _choleskyLower(NDArray matrix) {
    final n = matrix.shape[0];

    for (int i = 0; i < n; i++) {
      for (int j = 0; j <= i; j++) {
        if (i == j) {
          // Diagonal element
          double sum = 0.0;
          for (int k = 0; k < j; k++) {
            sum += matrix[[j, k]] * matrix[[j, k]];
          }
          final diagonal = matrix[[j, j]] - sum;
          if (diagonal <= 0.0) {
            throw LinAlgError(
              '${j + 1}-th leading minor is not positive definite',
            );
          }
          matrix[[j, j]] = math.sqrt(diagonal);
        } else {
          // Off-diagonal element
          double sum = 0.0;
          for (int k = 0; k < j; k++) {
            sum += matrix[[i, k]] * matrix[[j, k]];
          }
          matrix[[i, j]] = (matrix[[i, j]] - sum) / matrix[[j, j]];
        }
      }

      // Zero out upper triangle
      for (int j = i + 1; j < n; j++) {
        matrix[[i, j]] = 0.0;
      }
    }
  }

  /// Performs in-place Cholesky decomposition for upper triangular matrix.
  ///
  /// Computes U such that A = U^T U where U is upper triangular.
  static void _choleskyUpper(NDArray matrix) {
    final n = matrix.shape[0];

    for (int i = 0; i < n; i++) {
      for (int j = i; j < n; j++) {
        if (i == j) {
          // Diagonal element
          double sum = 0.0;
          for (int k = 0; k < i; k++) {
            sum += matrix[[k, i]] * matrix[[k, i]];
          }
          final diagonal = matrix[[i, i]] - sum;
          if (diagonal <= 0.0) {
            throw LinAlgError(
              '${i + 1}-th leading minor is not positive definite',
            );
          }
          matrix[[i, i]] = math.sqrt(diagonal);
        } else {
          // Off-diagonal element
          double sum = 0.0;
          for (int k = 0; k < i; k++) {
            sum += matrix[[k, i]] * matrix[[k, j]];
          }
          matrix[[i, j]] = (matrix[[i, j]] - sum) / matrix[[i, i]];
        }
      }

      // Zero out lower triangle
      for (int j = 0; j < i; j++) {
        matrix[[i, j]] = 0.0;
      }
    }
  }

  /// Compute the Cholesky decomposition of a matrix (clean version).
  ///
  /// Returns the Cholesky decomposition, A = L L* or A = U* U
  /// of a Hermitian positive-definite matrix A. Unlike cho_factor,
  /// this zeros out the unused triangle of the result.
  ///
  /// Parameters:
  /// - [a]: Matrix to be decomposed (must be square and positive-definite)
  /// - [lower]: Whether to compute lower or upper triangular factorization
  ///
  /// Returns:
  /// - [NDArray] containing the Cholesky factor
  ///
  /// Example:
  /// ```dart
  /// final a = NDArray.fromList([[4.0, 2.0], [2.0, 3.0]]);
  /// final l = LinAlg.cholesky(a, lower: true);
  /// ```
  static NDArray cholesky(NDArray a, {bool lower = false}) {
    final factorization = choFactor(a, lower: lower);
    return factorization.factorization;
  }
}
