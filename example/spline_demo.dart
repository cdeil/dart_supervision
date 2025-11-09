import 'package:dart_supervision/dart_supervision.dart';

void main() {
  print('=== Scipy Spline Interpolation Demo ===\n');

  // Demo 1: Basic 2D curve interpolation (like in supervision)
  print('Demo 1: Object Tracking Trajectory Smoothing');
  print('Simulating smooth curve generation for object tracking visualization');

  // Simulate object detection centers over time (like in supervision)
  final trackingPointsX = [100.0, 120.0, 150.0, 180.0, 200.0, 220.0];
  final trackingPointsY = [200.0, 180.0, 160.0, 140.0, 120.0, 100.0];

  print('Original tracking points:');
  for (int i = 0; i < trackingPointsX.length; i++) {
    print('  Point $i: (${trackingPointsX[i]}, ${trackingPointsY[i]})');
  }

  // Create parametric spline representation
  final (spline, parameterValues) = splprep([
    trackingPointsX,
    trackingPointsY,
  ], s: 20.0);

  print('\nParameter values for curve:');
  for (int i = 0; i < parameterValues.length; i++) {
    print('  u[$i] = ${parameterValues[i].toStringAsFixed(3)}');
  }

  // Evaluate spline at 20 points for smooth visualization
  final u = linspace(0.0, 1.0, 20);
  final smoothCurve = splev(u, spline);

  print('\nSmooth interpolated curve (20 points):');
  for (int i = 0; i < smoothCurve[0].length; i++) {
    final x = smoothCurve[0][i];
    final y = smoothCurve[1][i];
    print(
      '  Smooth Point $i: (${x.toStringAsFixed(1)}, ${y.toStringAsFixed(1)})',
    );
  }

  print('\n' + '=' * 60 + '\n');

  // Demo 2: 3D curve interpolation
  print('Demo 2: 3D Curve Interpolation');
  print('Creating a smooth 3D curve from sparse control points');

  final x3d = [0.0, 1.0, 2.0, 3.0];
  final y3d = [0.0, 1.0, 0.0, -1.0];
  final z3d = [0.0, 0.0, 1.0, 0.0];

  print('3D control points:');
  for (int i = 0; i < x3d.length; i++) {
    print('  Point $i: (${x3d[i]}, ${y3d[i]}, ${z3d[i]})');
  }

  final (spline3d, _) = splprep([x3d, y3d, z3d]);
  final u3d = linspace(0.0, 1.0, 10);
  final smooth3d = splev(u3d, spline3d);

  print('\nSmooth 3D curve (10 points):');
  for (int i = 0; i < smooth3d[0].length; i++) {
    final x = smooth3d[0][i];
    final y = smooth3d[1][i];
    final z = smooth3d[2][i];
    print(
      '  Point $i: (${x.toStringAsFixed(2)}, ${y.toStringAsFixed(2)}, ${z.toStringAsFixed(2)})',
    );
  }

  print('\n' + '=' * 60 + '\n');

  // Demo 3: Cubic spline comparison
  print('Demo 3: Cubic Spline Direct Usage');
  print('Demonstrating 1D cubic spline interpolation');

  final xData = [0.0, 1.0, 2.0, 3.0, 4.0];
  final yData = [0.0, 1.0, 4.0, 9.0, 16.0]; // y = x^2

  print('Original data points (y = x²):');
  for (int i = 0; i < xData.length; i++) {
    print('  (${xData[i]}, ${yData[i]})');
  }

  final cubicSpline = CubicSpline.fromPoints(xData, yData);
  final evalPoints = [0.5, 1.5, 2.5, 3.5];
  final interpolated = cubicSpline.eval(evalPoints);

  print('\nInterpolated values:');
  for (int i = 0; i < evalPoints.length; i++) {
    final x = evalPoints[i];
    final y = interpolated[i];
    final expected = x * x; // True value for y = x²
    final error = (y - expected).abs();
    print(
      '  x=${x.toStringAsFixed(1)}: interpolated=${y.toStringAsFixed(3)}, expected=${expected.toStringAsFixed(3)}, error=${error.toStringAsFixed(3)}',
    );
  }

  print('\n' + '=' * 60 + '\n');

  // Demo 4: Supervision-style usage for trace annotation
  print('Demo 4: Supervision TraceAnnotator Style Usage');
  print('Exactly like supervision\'s smooth trace generation');

  // This mimics the exact usage in supervision's TraceAnnotator
  final xy = [
    [50.0, 75.0, 100.0, 125.0, 150.0, 175.0], // x coordinates
    [300.0, 280.0, 260.0, 240.0, 220.0, 200.0], // y coordinates
  ];

  print('Object detection trajectory (${xy[0].length} points):');
  for (int i = 0; i < xy[0].length; i++) {
    print('  Detection $i: (${xy[0][i]}, ${xy[1][i]})');
  }

  // Check if we have enough points for smoothing (supervision checks len(xy) > 3)
  if (xy[0].length > 3) {
    print('\nApplying smoothing (like supervision with s=20)...');

    // This matches supervision's exact usage:
    // x, y = xy[:, 0], xy[:, 1]
    // tck, u = splprep([x, y], s=20)
    // x_new, y_new = splev(np.linspace(0, 1, 100), tck)
    final x = xy[0];
    final y = xy[1];
    final (tck, u) = splprep([x, y], s: 20);
    final uNew = linspace(0, 1, 100);
    final smoothResult = splev(uNew, tck);
    final xNew = smoothResult[0];
    final yNew = smoothResult[1];

    print('Generated ${xNew.length} smooth points for drawing');
    print('First 5 smooth points:');
    for (int i = 0; i < 5; i++) {
      print(
        '  Smooth $i: (${xNew[i].toStringAsFixed(1)}, ${yNew[i].toStringAsFixed(1)})',
      );
    }

    print('Last 5 smooth points:');
    for (int i = xNew.length - 5; i < xNew.length; i++) {
      print(
        '  Smooth $i: (${xNew[i].toStringAsFixed(1)}, ${yNew[i].toStringAsFixed(1)})',
      );
    }

    // Convert to integer coordinates like supervision does
    print('\nConverting to integer coordinates for drawing (like OpenCV)...');
    final splinePoints = <List<int>>[];
    for (int i = 0; i < xNew.length; i++) {
      splinePoints.add([xNew[i].round(), yNew[i].round()]);
    }

    print(
      'Ready for cv2.polylines() equivalent: ${splinePoints.length} integer points',
    );
    print('Sample integer points: ${splinePoints.take(5).toList()}');
  } else {
    print('Not enough points for smoothing (need > 3), using original points');
  }

  print('\n🎉 All demos completed successfully!');
  print(
    'The spline interpolation is ready for use in computer vision applications.',
  );
}
