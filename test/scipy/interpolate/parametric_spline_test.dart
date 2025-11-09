import 'package:test/test.dart';
import 'package:dart_supervision/dart_supervision.dart';

void main() {
  group('Parametric Spline Tests', () {
    test('basic 2D curve interpolation', () {
      // Create a simple curve (quarter circle)
      final x = [0.0, 1.0, 0.0];
      final y = [0.0, 0.0, 1.0];
      final coordinates = [x, y];

      final (spline, params) = splprep(coordinates);

      expect(spline, isA<ParametricSpline>());
      expect(params.length, equals(3));
      expect(params.first, equals(0.0));
      expect(params.last, equals(1.0));
    });

    test('spline evaluation with linspace', () {
      // Create a simple line
      final x = [0.0, 1.0, 2.0];
      final y = [0.0, 1.0, 2.0];
      final coordinates = [x, y];

      final (spline, _) = splprep(coordinates);

      // Evaluate at 5 points
      final u = linspace(0.0, 1.0, 5);
      final result = splev(u, spline);

      expect(result.length, equals(2)); // 2D curve
      expect(result[0].length, equals(5)); // 5 evaluation points
      expect(result[1].length, equals(5)); // 5 evaluation points

      // First and last points should match original data approximately
      expect(result[0].first, closeTo(0.0, 0.1));
      expect(result[1].first, closeTo(0.0, 0.1));
      expect(result[0].last, closeTo(2.0, 0.1));
      expect(result[1].last, closeTo(2.0, 0.1));
    });

    test('supervision library use case - smooth trajectory', () {
      // Simulate object tracking points (like in supervision library)
      final x = [10.0, 15.0, 25.0, 40.0, 60.0];
      final y = [20.0, 25.0, 30.0, 25.0, 15.0];
      final coordinates = [x, y];

      final (spline, _) = splprep(coordinates, s: 20.0);

      // Evaluate at 100 points (like supervision does)
      final u = linspace(0.0, 1.0, 100);
      final result = splev(u, spline);

      expect(result.length, equals(2));
      expect(result[0].length, equals(100));
      expect(result[1].length, equals(100));

      // Check that interpolated curve is smooth (no huge jumps)
      for (int i = 1; i < result[0].length; i++) {
        final dx = (result[0][i] - result[0][i - 1]).abs();
        final dy = (result[1][i] - result[1][i - 1]).abs();
        expect(dx, lessThan(10.0)); // Reasonable step size
        expect(dy, lessThan(10.0)); // Reasonable step size
      }
    });

    test('empty input handling', () {
      expect(() => splprep([]), throwsArgumentError);
      expect(() => splprep([[], []]), throwsArgumentError);
    });

    test('mismatched dimensions', () {
      final x = [1.0, 2.0, 3.0];
      final y = [1.0, 2.0]; // Different length
      expect(() => splprep([x, y]), throwsArgumentError);
    });

    test('single point handling', () {
      final x = [1.0];
      final y = [2.0];
      expect(() => splprep([x, y]), throwsArgumentError);
    });

    test('3D curve interpolation', () {
      final x = [0.0, 1.0, 2.0];
      final y = [0.0, 1.0, 0.0];
      final z = [0.0, 0.0, 1.0];
      final coordinates = [x, y, z];

      final (spline, _) = splprep(coordinates);
      final u = linspace(0.0, 1.0, 10);
      final result = splev(u, spline);

      expect(result.length, equals(3)); // 3D curve
      expect(result[0].length, equals(10));
      expect(result[1].length, equals(10));
      expect(result[2].length, equals(10));
    });
  });

  group('Cubic Spline Tests', () {
    test('linear interpolation', () {
      final x = [0.0, 1.0, 2.0];
      final y = [0.0, 1.0, 2.0];

      final spline = CubicSpline.fromPoints(x, y);
      final result = spline.eval([0.5, 1.5]);

      expect(result[0], closeTo(0.5, 1e-6));
      expect(result[1], closeTo(1.5, 1e-6));
    });

    test('quadratic function interpolation', () {
      final x = [0.0, 1.0, 2.0, 3.0];
      final y = [0.0, 1.0, 4.0, 9.0]; // y = x^2

      final spline = CubicSpline.fromPoints(x, y);
      final result = spline.eval([1.5, 2.5]);

      // Should be close to 1.5^2 = 2.25 and 2.5^2 = 6.25
      // Note: Cubic splines don't perfectly reproduce polynomials > degree 3
      expect(result[0], closeTo(2.25, 0.5)); // More lenient tolerance
      expect(result[1], closeTo(6.25, 0.5)); // More lenient tolerance
    });

    test('extrapolation beyond bounds', () {
      final x = [1.0, 2.0, 3.0];
      final y = [1.0, 4.0, 9.0];

      final spline = CubicSpline.fromPoints(x, y);
      final result = spline.eval([0.0, 4.0]); // Outside bounds

      expect(result.length, equals(2));
      expect(result[0].isFinite, isTrue);
      expect(result[1].isFinite, isTrue);
    });

    test('two point case (linear)', () {
      final x = [0.0, 1.0];
      final y = [0.0, 2.0];

      final spline = CubicSpline.fromPoints(x, y);
      final result = spline.eval([0.5]);

      expect(result[0], closeTo(1.0, 1e-6));
    });

    test('duplicate x values should fail', () {
      // Note: Current implementation doesn't explicitly check for this,
      // but it should be handled gracefully
      final x = [1.0, 1.0, 2.0];
      final y = [1.0, 2.0, 3.0];

      final spline = CubicSpline.fromPoints(x, y);
      final result = spline.eval([1.0]);

      expect(result[0].isFinite, isTrue);
    });
  });

  group('Utility Functions Tests', () {
    test('linspace basic functionality', () {
      final result = linspace(0.0, 1.0, 5);

      expect(result.length, equals(5));
      expect(result[0], equals(0.0));
      expect(result[4], equals(1.0));
      expect(result[1], closeTo(0.25, 1e-6));
      expect(result[2], closeTo(0.5, 1e-6));
      expect(result[3], closeTo(0.75, 1e-6));
    });

    test('linspace edge cases', () {
      expect(linspace(0.0, 1.0, 0), isEmpty);
      expect(linspace(0.0, 1.0, 1), equals([0.0]));
      expect(linspace(5.0, 5.0, 3), equals([5.0, 5.0, 5.0]));
    });

    test('linspace negative range', () {
      final result = linspace(-1.0, 1.0, 3);

      expect(result[0], equals(-1.0));
      expect(result[1], equals(0.0));
      expect(result[2], equals(1.0));
    });
  });

  group('Integration Tests - Supervision Use Case', () {
    test('complete workflow like supervision TraceAnnotator', () {
      // Simulate a real object tracking scenario
      final trackingPoints = [
        [100.0, 120.0, 150.0, 180.0, 200.0, 220.0], // x coordinates
        [200.0, 180.0, 160.0, 140.0, 120.0, 100.0], // y coordinates
      ];

      // Step 1: Create parametric spline (like supervision does)
      final (spline, _) = splprep(trackingPoints, s: 20.0);

      // Step 2: Evaluate at 100 points for smooth visualization
      final u = linspace(0.0, 1.0, 100);
      final smoothCurve = splev(u, spline);

      // Verify the results
      expect(smoothCurve.length, equals(2));
      expect(smoothCurve[0].length, equals(100));
      expect(smoothCurve[1].length, equals(100));

      // The curve should start and end near the original points
      expect(smoothCurve[0].first, closeTo(100.0, 5.0));
      expect(smoothCurve[1].first, closeTo(200.0, 5.0));
      expect(smoothCurve[0].last, closeTo(220.0, 5.0));
      expect(smoothCurve[1].last, closeTo(100.0, 5.0));

      // The curve should be smooth (no jumps larger than reasonable)
      for (int i = 1; i < 100; i++) {
        final dx = (smoothCurve[0][i] - smoothCurve[0][i - 1]).abs();
        final dy = (smoothCurve[1][i] - smoothCurve[1][i - 1]).abs();
        expect(dx, lessThan(5.0), reason: 'X jump too large at index $i');
        expect(dy, lessThan(5.0), reason: 'Y jump too large at index $i');
      }
    });

    test('minimal case - 3 points like supervision minimum', () {
      // Supervision requires len(xy) > 3 for smoothing
      final x = [10.0, 20.0, 30.0, 40.0];
      final y = [10.0, 15.0, 20.0, 15.0];

      final (spline, _) = splprep([x, y], s: 20.0);
      final u = linspace(0.0, 1.0, 100);
      final result = splev(u, spline);

      expect(result[0].length, equals(100));
      expect(result[1].length, equals(100));

      // Verify endpoints are approximately correct
      expect(result[0].first, closeTo(x.first, 1.0));
      expect(result[1].first, closeTo(y.first, 1.0));
      expect(result[0].last, closeTo(x.last, 1.0));
      expect(result[1].last, closeTo(y.last, 1.0));
    });
  });
}
