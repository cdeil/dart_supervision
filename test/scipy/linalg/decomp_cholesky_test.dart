import 'package:test/test.dart';
import 'package:dart_supervision/dart_supervision.dart';
import 'dart:math' as math;

void main() {
  group('Cholesky Decomposition', () {
    test('cho_factor computes correct decomposition for simple 2x2 matrix', () {
      // Test case from SciPy documentation
      final a = NDArray.fromList([
        [4.0, 2.0],
        [2.0, 3.0],
      ]);

      final result = LinAlg.choFactor(a, lower: true);
      final l = result.factorization;

      expect(result.lower, isTrue);
      expect(l.shape, equals([2, 2]));

      // Verify L * L^T = A
      final lt = l.transpose();
      final reconstructed = l.dot(lt);

      expect(reconstructed[[0, 0]], closeTo(a[[0, 0]], 1e-10));
      expect(reconstructed[[0, 1]], closeTo(a[[0, 1]], 1e-10));
      expect(reconstructed[[1, 0]], closeTo(a[[1, 0]], 1e-10));
      expect(reconstructed[[1, 1]], closeTo(a[[1, 1]], 1e-10));

      // Check that upper triangle is zeroed
      expect(l[[0, 1]], equals(0.0));
    });

    test('cho_factor computes correct upper triangular decomposition', () {
      final a = NDArray.fromList([
        [4.0, 2.0],
        [2.0, 3.0],
      ]);

      final result = LinAlg.choFactor(a, lower: false);
      final u = result.factorization;

      expect(result.lower, isFalse);
      expect(u.shape, equals([2, 2]));

      // Verify U^T * U = A
      final ut = u.transpose();
      final reconstructed = ut.dot(u);

      expect(reconstructed[[0, 0]], closeTo(a[[0, 0]], 1e-10));
      expect(reconstructed[[0, 1]], closeTo(a[[0, 1]], 1e-10));
      expect(reconstructed[[1, 0]], closeTo(a[[1, 0]], 1e-10));
      expect(reconstructed[[1, 1]], closeTo(a[[1, 1]], 1e-10));

      // Check that lower triangle is zeroed
      expect(u[[1, 0]], equals(0.0));
    });

    test('cho_factor handles 3x3 matrix correctly', () {
      final a = NDArray.fromList([
        [8.0, 2.0, 3.0],
        [2.0, 9.0, 3.0],
        [3.0, 3.0, 6.0],
      ]);

      final result = LinAlg.choFactor(a, lower: true);
      final l = result.factorization;

      // Verify L * L^T = A
      final lt = l.transpose();
      final reconstructed = l.dot(lt);

      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          expect(
            reconstructed[[i, j]],
            closeTo(a[[i, j]], 1e-10),
            reason: 'Element [$i, $j] mismatch',
          );
        }
      }
    });

    test('cho_factor throws error for non-positive definite matrix', () {
      final a = NDArray.fromList([
        [1.0, 2.0],
        [2.0, 1.0], // This matrix has determinant 1-4 = -3 < 0
      ]);

      expect(() => LinAlg.choFactor(a), throwsA(isA<LinAlgError>()));
    });

    test('cho_factor throws error for non-square matrix', () {
      final a = NDArray.fromList([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
      ]);

      expect(() => LinAlg.choFactor(a), throwsArgumentError);
    });

    test('cho_factor handles empty matrix', () {
      final a = NDArray([0, 0]);
      final result = LinAlg.choFactor(a);

      expect(result.factorization.shape, equals([0, 0]));
    });

    test('cho_factor handles 1x1 matrix', () {
      final a = NDArray.fromList([
        [4.0],
      ]);
      final result = LinAlg.choFactor(a, lower: true);

      expect(result.factorization[[0, 0]], closeTo(2.0, 1e-10));

      // Verify reconstruction
      final l = result.factorization;
      final reconstructed = l.dot(l.transpose());
      expect(reconstructed[[0, 0]], closeTo(4.0, 1e-10));
    });
  });

  group('Cholesky Solve', () {
    test('cho_solve solves simple 2x2 system correctly', () {
      final a = NDArray.fromList([
        [4.0, 2.0],
        [2.0, 3.0],
      ]);
      final b = NDArray.fromList([8.0, 7.0]);

      final chol = LinAlg.choFactor(a, lower: true);
      final x = LinAlg.choSolve(chol, b);

      expect(x.shape, equals([2]));

      // Verify A * x = b
      final ax = a.dot(x.reshape([2, 1])).flatten();
      expect(ax[[0]], closeTo(b[[0]], 1e-10));
      expect(ax[[1]], closeTo(b[[1]], 1e-10));
    });

    test('cho_solve works with upper triangular factorization', () {
      final a = NDArray.fromList([
        [4.0, 2.0],
        [2.0, 3.0],
      ]);
      final b = NDArray.fromList([8.0, 7.0]);

      final chol = LinAlg.choFactor(a, lower: false);
      final x = LinAlg.choSolve(chol, b);

      // Verify A * x = b
      final ax = a.dot(x.reshape([2, 1])).flatten();
      expect(ax[[0]], closeTo(b[[0]], 1e-10));
      expect(ax[[1]], closeTo(b[[1]], 1e-10));
    });

    test('cho_solve handles multiple right-hand sides', () {
      final a = NDArray.fromList([
        [4.0, 2.0],
        [2.0, 3.0],
      ]);
      final b = NDArray.fromList([
        [8.0, 6.0],
        [7.0, 5.0],
      ]);

      final chol = LinAlg.choFactor(a, lower: true);
      final x = LinAlg.choSolve(chol, b);

      expect(x.shape, equals([2, 2]));

      // Verify A * x = b for each column
      final ax = a.dot(x);
      for (int col = 0; col < 2; col++) {
        for (int row = 0; row < 2; row++) {
          expect(
            ax[[row, col]],
            closeTo(b[[row, col]], 1e-10),
            reason: 'Column $col, row $row mismatch',
          );
        }
      }
    });

    test('cho_solve handles 3x3 system', () {
      final a = NDArray.fromList([
        [8.0, 2.0, 3.0],
        [2.0, 9.0, 3.0],
        [3.0, 3.0, 6.0],
      ]);
      final b = NDArray.fromList([1.0, 1.0, 1.0]);

      final chol = LinAlg.choFactor(a, lower: true);
      final x = LinAlg.choSolve(chol, b);

      expect(x.shape, equals([3]));

      // Verify A * x = b
      final ax = a.dot(x.reshape([3, 1])).flatten();
      for (int i = 0; i < 3; i++) {
        expect(ax[[i]], closeTo(b[[i]], 1e-10), reason: 'Element $i mismatch');
      }
    });

    test('cho_solve validates input dimensions', () {
      final a = NDArray.fromList([
        [4.0, 2.0],
        [2.0, 3.0],
      ]);
      final badB = NDArray.fromList([1.0, 2.0, 3.0]); // Wrong size

      final chol = LinAlg.choFactor(a);

      expect(() => LinAlg.choSolve(chol, badB), throwsArgumentError);
    });

    test('cho_solve handles empty system', () {
      final a = NDArray([0, 0]);
      final b = NDArray([0]);

      final chol = LinAlg.choFactor(a);
      final x = LinAlg.choSolve(chol, b);

      expect(x.shape, equals([0]));
    });

    test('cho_solve from SciPy documentation example', () {
      // Example from SciPy documentation
      final a = NDArray.fromList([
        [9.0, 3.0, 1.0, 5.0],
        [3.0, 7.0, 5.0, 1.0],
        [1.0, 5.0, 9.0, 2.0],
        [5.0, 1.0, 2.0, 6.0],
      ]);
      final b = NDArray.fromList([1.0, 1.0, 1.0, 1.0]);

      final chol = LinAlg.choFactor(a);
      final x = LinAlg.choSolve(chol, b);

      // Verify A * x = b
      final ax = a.dot(x.reshape([4, 1])).flatten();
      for (int i = 0; i < 4; i++) {
        expect(ax[[i]], closeTo(b[[i]], 1e-10), reason: 'Element $i mismatch');
      }
    });
  });

  group('Cholesky Function', () {
    test('cholesky produces clean triangular matrix', () {
      final a = NDArray.fromList([
        [4.0, 2.0],
        [2.0, 3.0],
      ]);

      final l = LinAlg.cholesky(a, lower: true);

      expect(l.shape, equals([2, 2]));
      expect(l[[0, 1]], equals(0.0)); // Upper triangle should be zero

      // Verify L * L^T = A
      final lt = l.transpose();
      final reconstructed = l.dot(lt);

      expect(reconstructed[[0, 0]], closeTo(a[[0, 0]], 1e-10));
      expect(reconstructed[[0, 1]], closeTo(a[[0, 1]], 1e-10));
      expect(reconstructed[[1, 0]], closeTo(a[[1, 0]], 1e-10));
      expect(reconstructed[[1, 1]], closeTo(a[[1, 1]], 1e-10));
    });

    test('cholesky upper triangular', () {
      final a = NDArray.fromList([
        [4.0, 2.0],
        [2.0, 3.0],
      ]);

      final u = LinAlg.cholesky(a, lower: false);

      expect(u.shape, equals([2, 2]));
      expect(u[[1, 0]], equals(0.0)); // Lower triangle should be zero

      // Verify U^T * U = A
      final ut = u.transpose();
      final reconstructed = ut.dot(u);

      expect(reconstructed[[0, 0]], closeTo(a[[0, 0]], 1e-10));
      expect(reconstructed[[0, 1]], closeTo(a[[0, 1]], 1e-10));
      expect(reconstructed[[1, 0]], closeTo(a[[1, 0]], 1e-10));
      expect(reconstructed[[1, 1]], closeTo(a[[1, 1]], 1e-10));
    });
  });

  group('Error Handling', () {
    test('LinAlgError has proper message', () {
      final error = LinAlgError('Test error message');
      expect(error.toString(), contains('Test error message'));
    });

    test('cho_factor handles singular matrix gracefully', () {
      final a = NDArray.fromList([
        [1.0, 1.0],
        [1.0, 1.0], // Rank deficient matrix
      ]);

      expect(() => LinAlg.choFactor(a), throwsA(isA<LinAlgError>()));
    });

    test('cho_factor validates 2D input', () {
      final a = NDArray.fromList([1.0, 2.0, 3.0]); // 1D array

      expect(() => LinAlg.choFactor(a), throwsArgumentError);
    });
  });

  group('Numerical Accuracy', () {
    test('maintains accuracy for larger matrices', () {
      // Create a random symmetric positive definite matrix
      final n = 5;
      final random = math.Random(42); // Fixed seed for reproducibility

      // Generate random matrix
      final temp = NDArray([n, n]);
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
          temp[[i, j]] = random.nextDouble() - 0.5;
        }
      }

      // Make it symmetric positive definite: A = B^T * B + I
      final a = temp.transpose().dot(temp) + NDArray.eye(n);

      final chol = LinAlg.choFactor(a, lower: true);
      final l = chol.factorization;

      // Verify reconstruction
      final reconstructed = l.dot(l.transpose());
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
          expect(
            reconstructed[[i, j]],
            closeTo(a[[i, j]], 1e-8),
            reason: 'Element [$i, $j] accuracy issue',
          );
        }
      }
    });

    test('solve maintains accuracy for larger systems', () {
      final n = 4;
      final a = NDArray([n, n]);

      // Create a well-conditioned symmetric positive definite matrix
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
          if (i == j) {
            a[[i, j]] = 10.0; // Strong diagonal
          } else if ((i - j).abs() == 1) {
            a[[i, j]] = 1.0; // Weak off-diagonal
          } else {
            a[[i, j]] = 0.0;
          }
        }
      }

      final b = NDArray.ones([n]);
      final chol = LinAlg.choFactor(a, lower: true);
      final x = LinAlg.choSolve(chol, b);

      // Verify solution
      final ax = a.dot(x.reshape([n, 1])).flatten();
      for (int i = 0; i < n; i++) {
        expect(
          ax[[i]],
          closeTo(b[[i]], 1e-12),
          reason: 'Solution accuracy issue at element $i',
        );
      }
    });
  });
}
