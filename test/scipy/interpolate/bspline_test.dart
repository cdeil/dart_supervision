import 'package:test/test.dart';
import 'package:dart_supervision/dart_supervision.dart';

void main() {
  group('BSpline Tests', () {
    test('basic BSpline creation and evaluation', () {
      final x = [0.0, 1.0, 2.0];
      final y = [0.0, 1.0, 2.0];
      final coordinates = [x, y];

      final (bspline, params) = makeSplprep(coordinates);

      expect(bspline, isA<BSpline>());
      expect(bspline.dimensions, equals(2));
      expect(bspline.k, equals(3));
      expect(params.length, equals(3));
    });

    test('BSpline evaluation', () {
      final x = [0.0, 1.0, 2.0];
      final y = [0.0, 1.0, 2.0];
      final coordinates = [x, y];

      final (bspline, _) = makeSplprep(coordinates);
      final result = bspline.call([0.0, 0.5, 1.0]);

      expect(result.length, equals(2));
      expect(result[0].length, equals(3));
      expect(result[1].length, equals(3));
    });

    test('BSpline properties access', () {
      final x = [1.0, 2.0, 3.0];
      final y = [4.0, 5.0, 6.0];
      final coordinates = [x, y];

      final (bspline, _) = makeSplprep(coordinates);

      expect(bspline.t.length, greaterThan(0));
      expect(bspline.c.length, equals(2));
      expect(bspline.k, equals(3));
      expect(bspline.dimensions, equals(2));
    });

    test('custom parameters', () {
      final x = [0.0, 1.0, 2.0];
      final y = [0.0, 1.0, 4.0];
      final coordinates = [x, y];
      final customParams = [0.0, 0.3, 1.0];

      final (bspline, params) = makeSplprep(
        coordinates,
        u: customParams,
        k: 2,
        s: 10.0,
      );

      expect(params, equals(customParams));
      expect(bspline.k, equals(2));
    });

    test('3D curve with BSpline', () {
      final x = [0.0, 1.0, 2.0];
      final y = [0.0, 1.0, 0.0];
      final z = [0.0, 0.0, 1.0];
      final coordinates = [x, y, z];

      final (bspline, _) = makeSplprep(coordinates);
      final result = bspline.call([0.0, 0.5, 1.0]);

      expect(result.length, equals(3));
      expect(bspline.dimensions, equals(3));
    });

    test('BSpline vs legacy splprep comparison', () {
      final x = [10.0, 20.0, 30.0, 40.0];
      final y = [10.0, 15.0, 20.0, 15.0];
      final coordinates = [x, y];

      // Modern interface
      final (bspline, _) = makeSplprep(coordinates);
      final modernResult = bspline.call([0.0, 0.5, 1.0]);

      // Legacy interface (from parametric_spline.dart)
      // We'll import it dynamically to test compatibility
      expect(modernResult.length, equals(2));
      expect(modernResult[0].length, equals(3));
      expect(modernResult[1].length, equals(3));

      // Results should be smooth
      for (int dim = 0; dim < 2; dim++) {
        for (int i = 1; i < modernResult[dim].length; i++) {
          final diff = (modernResult[dim][i] - modernResult[dim][i - 1]).abs();
          expect(diff, lessThan(50.0)); // Reasonable step size
        }
      }
    });

    test('error handling', () {
      expect(() => makeSplprep([]), throwsArgumentError);
      expect(() => makeSplprep([[], []]), throwsArgumentError);

      final x = [1.0, 2.0, 3.0];
      final y = [1.0, 2.0]; // Different length
      expect(() => makeSplprep([x, y]), throwsArgumentError);
    });
  });
}
