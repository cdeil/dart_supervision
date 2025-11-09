import 'package:test/test.dart';
import 'package:dart_supervision/dart_supervision.dart' as sv;

void main() {
  group('NDArray Creation', () {
    test('creates zeros array', () {
      final arr = sv.NDArray.zeros([2, 3]);
      expect(arr.shape, equals([2, 3]));
      expect(arr.size, equals(6));
      expect(arr.ndim, equals(2));
      expect(arr[[0, 0]], equals(0.0));
      expect(arr[[1, 2]], equals(0.0));
    });

    test('creates ones array', () {
      final arr = sv.NDArray.ones([2, 2]);
      expect(arr.shape, equals([2, 2]));
      expect(arr[[0, 0]], equals(1.0));
      expect(arr[[1, 1]], equals(1.0));
    });

    test('creates array from list', () {
      final data = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
      ];
      final arr = sv.NDArray.fromList(data);

      expect(arr.shape, equals([2, 3]));
      expect(arr[[0, 0]], equals(1.0));
      expect(arr[[0, 1]], equals(2.0));
      expect(arr[[1, 2]], equals(6.0));
    });

    test('creates identity matrix', () {
      final eye = sv.NDArray.eye(3);
      expect(eye.shape, equals([3, 3]));
      expect(eye[[0, 0]], equals(1.0));
      expect(eye[[1, 1]], equals(1.0));
      expect(eye[[2, 2]], equals(1.0));
      expect(eye[[0, 1]], equals(0.0));
      expect(eye[[1, 0]], equals(0.0));
    });

    test('creates arange array', () {
      final arr = sv.NDArray.arange(0, 5);
      expect(arr.shape, equals([5]));
      expect(arr[[0]], equals(0.0));
      expect(arr[[4]], equals(4.0));

      final arr2 = sv.NDArray.arange(1, 10, 2);
      expect(arr2.shape, equals([5]));
      expect(arr2[[0]], equals(1.0));
      expect(arr2[[4]], equals(9.0));
    });

    test('creates linspace array', () {
      final arr = sv.NDArray.linspace(0, 1, 5);
      expect(arr.shape, equals([5]));
      expect(arr[[0]], equals(0.0));
      expect(arr[[4]], equals(1.0));
      expect(arr[[2]], closeTo(0.5, 1e-10));
    });
  });

  group('NDArray Properties', () {
    test('computes strides correctly', () {
      final arr2d = sv.NDArray.zeros([3, 4]);
      final arr3d = sv.NDArray.zeros([2, 3, 4]);

      expect(arr2d.strides, equals([4, 1]));
      expect(arr3d.strides, equals([12, 4, 1]));
    });

    test('has correct size and shape', () {
      final arr = sv.NDArray.zeros([2, 3, 4]);
      expect(arr.shape, equals([2, 3, 4]));
      expect(arr.size, equals(24));
      expect(arr.ndim, equals(3));
    });

    test('has correct data type info', () {
      final arr = sv.NDArray.zeros([2, 2]);
      expect(arr.dtype, equals(double));
      expect(arr.itemsize, equals(8)); // double = 64 bits = 8 bytes
      expect(arr.nbytes, equals(32)); // 4 elements * 8 bytes each
    });
  });

  group('NDArray Indexing', () {
    test('supports element access and assignment', () {
      final arr = sv.NDArray.zeros([3, 3]);

      arr[[0, 1]] = 10.0;
      arr[[1, 0]] = 20.0;
      arr[[2, 2]] = 30.0;

      expect(arr[[0, 0]], equals(0.0));
      expect(arr[[0, 1]], equals(10.0));
      expect(arr[[1, 0]], equals(20.0));
      expect(arr[[2, 2]], equals(30.0));
    });

    test('validates indices bounds', () {
      final arr = sv.NDArray.zeros([2, 3]);

      expect(() => arr[[2, 0]], throwsRangeError);
      expect(() => arr[[0, 3]], throwsRangeError);
      expect(() => arr[[-1, 0]], throwsRangeError);
    });

    test('validates indices dimensions', () {
      final arr = sv.NDArray.zeros([2, 3]);

      expect(() => arr[[0]], throwsArgumentError);
      expect(() => arr[[0, 1, 2]], throwsArgumentError);
    });
  });

  group('NDArray Reshaping', () {
    test('reshapes array preserving data', () {
      final arr = sv.NDArray.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
      final reshaped = arr.reshape([2, 3]);

      expect(reshaped.shape, equals([2, 3]));
      expect(reshaped.size, equals(6));
      expect(reshaped[[0, 0]], equals(1.0));
      expect(reshaped[[1, 2]], equals(6.0));
    });

    test('validates reshape size compatibility', () {
      final arr = sv.NDArray.zeros([2, 3]);

      expect(() => arr.reshape([2, 4]), throwsArgumentError);
      expect(() => arr.reshape([7]), throwsArgumentError);
    });

    test('transposes 2D arrays', () {
      final arr = sv.NDArray.fromList([
        [1.0, 2.0],
        [3.0, 4.0],
      ]);
      final transposed = arr.transpose();

      expect(transposed.shape, equals([2, 2]));
      expect(transposed[[0, 0]], equals(1.0));
      expect(transposed[[0, 1]], equals(3.0));
      expect(transposed[[1, 0]], equals(2.0));
      expect(transposed[[1, 1]], equals(4.0));
    });

    test('flattens arrays', () {
      final arr = sv.NDArray.fromList([
        [1.0, 2.0],
        [3.0, 4.0],
      ]);
      final flattened = arr.flatten();

      expect(flattened.shape, equals([4]));
      expect(flattened[[0]], equals(1.0));
      expect(flattened[[3]], equals(4.0));
    });
  });

  group('NDArray Boolean Indexing', () {
    test('creates boolean masks', () {
      final arr = sv.NDArray.fromList([1.0, 2.0, 3.0, 4.0]);

      final mask1 = arr > 2.0;
      final mask2 = arr <= 3.0;

      expect(mask1[[0]], equals(0.0)); // false
      expect(mask1[[3]], equals(1.0)); // true
      expect(mask2[[0]], equals(1.0)); // true
      expect(mask2[[3]], equals(0.0)); // false
    });

    test('applies boolean masks', () {
      final arr = sv.NDArray.fromList([1.0, 2.0, 3.0, 4.0, 5.0]);
      final mask = arr > 3.0;

      final filtered = arr & mask;
      expect(filtered.shape, equals([2])); // 4.0 and 5.0
      expect(filtered[[0]], equals(4.0));
      expect(filtered[[1]], equals(5.0));
    });

    test('validates mask shape', () {
      final arr = sv.NDArray.zeros([2, 3]);
      final badMask = sv.NDArray.zeros([2, 2]);

      expect(() => arr & badMask, throwsArgumentError);
    });
  });

  group('NDArray Arithmetic Operations', () {
    test('scalar addition', () {
      final arr = sv.NDArray.fromList([1.0, 2.0, 3.0]);
      final result = arr + 10;

      expect(result[[0]], equals(11.0));
      expect(result[[1]], equals(12.0));
      expect(result[[2]], equals(13.0));
    });

    test('scalar subtraction', () {
      final arr = sv.NDArray.fromList([10.0, 20.0, 30.0]);
      final result = arr - 5;

      expect(result[[0]], equals(5.0));
      expect(result[[1]], equals(15.0));
      expect(result[[2]], equals(25.0));
    });

    test('scalar multiplication', () {
      final arr = sv.NDArray.fromList([1.0, 2.0, 3.0]);
      final result = arr * 2;

      expect(result[[0]], equals(2.0));
      expect(result[[1]], equals(4.0));
      expect(result[[2]], equals(6.0));
    });

    test('scalar division', () {
      final arr = sv.NDArray.fromList([10.0, 20.0, 30.0]);
      final result = arr / 2;

      expect(result[[0]], equals(5.0));
      expect(result[[1]], equals(10.0));
      expect(result[[2]], equals(15.0));
    });

    test('element-wise array operations', () {
      final arr1 = sv.NDArray.fromList([1.0, 2.0, 3.0]);
      final arr2 = sv.NDArray.fromList([10.0, 20.0, 30.0]);

      final sum = arr1 + arr2;
      final diff = arr2 - arr1;
      final product = arr1 * arr2;
      final quotient = arr2 / arr1;

      expect(sum[[0]], equals(11.0));
      expect(sum[[2]], equals(33.0));

      expect(diff[[0]], equals(9.0));
      expect(diff[[2]], equals(27.0));

      expect(product[[1]], equals(40.0));
      expect(quotient[[1]], equals(10.0));
    });

    test('validates array shapes for operations', () {
      final arr1 = sv.NDArray.zeros([2, 3]);
      final arr2 = sv.NDArray.zeros([3, 2]);

      expect(() => arr1 + arr2, throwsArgumentError);
      expect(() => arr1 - arr2, throwsArgumentError);
    });
  });

  group('NDArray Matrix Operations', () {
    test('matrix multiplication', () {
      final a = sv.NDArray.fromList([
        [1.0, 2.0],
        [3.0, 4.0],
      ]);
      final b = sv.NDArray.fromList([
        [5.0, 6.0],
        [7.0, 8.0],
      ]);

      final result = a.dot(b);

      expect(result.shape, equals([2, 2]));
      expect(result[[0, 0]], equals(19.0)); // 1*5 + 2*7
      expect(result[[0, 1]], equals(22.0)); // 1*6 + 2*8
      expect(result[[1, 0]], equals(43.0)); // 3*5 + 4*7
      expect(result[[1, 1]], equals(50.0)); // 3*6 + 4*8
    });

    test('validates matrix dimensions', () {
      final a = sv.NDArray.zeros([2, 3]);
      final b = sv.NDArray.zeros([2, 2]);

      expect(() => a.dot(b), throwsArgumentError);
    });

    test('requires 2D arrays for dot product', () {
      final a = sv.NDArray.zeros([5]);
      final b = sv.NDArray.zeros([5]);

      expect(() => a.dot(b), throwsArgumentError);
    });
  });

  group('NDArray Aggregate Functions', () {
    test('computes sum', () {
      final arr = sv.NDArray.fromList([1.0, 2.0, 3.0, 4.0]);
      expect(arr.sum(), equals(10.0));

      final zeros = sv.NDArray.zeros([3, 3]);
      expect(zeros.sum(), equals(0.0));
    });

    test('computes mean', () {
      final arr = sv.NDArray.fromList([1.0, 2.0, 3.0, 4.0]);
      expect(arr.mean(), equals(2.5));

      final ones = sv.NDArray.ones([2, 5]);
      expect(ones.mean(), equals(1.0));
    });

    test('finds min and max', () {
      final arr = sv.NDArray.fromList([3.0, 1.0, 4.0, 1.0, 5.0]);

      expect(arr.min(), equals(1.0));
      expect(arr.max(), equals(5.0));
    });

    test('handles empty arrays', () {
      final empty = sv.NDArray([0]);

      expect(() => empty.min(), throwsStateError);
      expect(() => empty.max(), throwsStateError);
    });

    test('square root function', () {
      final arr = sv.NDArray.fromList([1.0, 4.0, 9.0, 16.0]);
      final result = arr.sqrt();

      expect(result[[0]], equals(1.0));
      expect(result[[1]], equals(2.0));
      expect(result[[2]], equals(3.0));
      expect(result[[3]], equals(4.0));
    });
  });

  group('NDArray Utility Functions', () {
    test('creates copy', () {
      final original = sv.NDArray.fromList([1.0, 2.0, 3.0]);
      final copy = original.copy();

      copy[[0]] = 99.0;

      expect(original[[0]], equals(1.0)); // Original unchanged
      expect(copy[[0]], equals(99.0));
    });

    test('converts to list', () {
      final arr1d = sv.NDArray.fromList([1.0, 2.0, 3.0]);
      final arr2d = sv.NDArray.fromList([
        [1.0, 2.0],
        [3.0, 4.0],
      ]);

      expect(arr1d.toList(), equals([1.0, 2.0, 3.0]));
      expect(
        arr2d.toList(),
        equals([
          [1.0, 2.0],
          [3.0, 4.0],
        ]),
      );
    });

    test('string representation', () {
      final small = sv.NDArray.fromList([1.0, 2.0, 3.0]);
      final large = sv.NDArray.zeros([100, 100]);

      expect(small.toString(), contains('NDArray'));
      expect(small.toString(), contains('shape'));
      expect(large.toString(), isNotNull);
    });

    test('equality comparison', () {
      final arr1 = sv.NDArray.fromList([1.0, 2.0, 3.0]);
      final arr2 = sv.NDArray.fromList([1.0, 2.0, 3.0]);
      final arr3 = sv.NDArray.fromList([1.0, 2.0, 4.0]);

      expect(arr1 == arr2, isTrue);
      expect(arr1 == arr3, isFalse);
    });

    test('hash code consistency', () {
      final arr1 = sv.NDArray.fromList([1.0, 2.0, 3.0]);
      final arr2 = sv.NDArray.fromList([1.0, 2.0, 3.0]);

      expect(arr1.hashCode == arr2.hashCode, isTrue);
    });
  });
}
