import 'package:dart_supervision/dart_supervision.dart';

void main() {
  print('=== Enhanced Scipy Spline Interpolation Demo ===\n');

  // Demo: Comparison between Legacy and Modern Interfaces
  print('🔄 Legacy vs Modern Interface Comparison');
  print('Testing both splprep/splev and makeSplprep/BSpline approaches');

  // Test data - object tracking trajectory
  final trackingX = [100.0, 120.0, 150.0, 180.0, 200.0];
  final trackingY = [200.0, 180.0, 160.0, 140.0, 120.0];
  final coordinates = [trackingX, trackingY];

  print('\nOriginal tracking data:');
  for (int i = 0; i < trackingX.length; i++) {
    print('  Point $i: (${trackingX[i]}, ${trackingY[i]})');
  }

  // ===== LEGACY INTERFACE =====
  print('\n📜 Using Legacy Interface (splprep/splev):');
  final (legacySpline, legacyParams) = splprep(coordinates, s: 20.0);
  final legacyU = linspace(0.0, 1.0, 10);
  final legacyResult = splev(legacyU, legacySpline);

  print('Legacy result (10 points):');
  for (int i = 0; i < legacyResult[0].length; i++) {
    final x = legacyResult[0][i];
    final y = legacyResult[1][i];
    print('  Legacy [$i]: (${x.toStringAsFixed(1)}, ${y.toStringAsFixed(1)})');
  }

  // ===== MODERN INTERFACE =====
  print('\n🆕 Using Modern Interface (makeSplprep/BSpline):');
  final (modernSpline, modernParams) = makeSplprep(coordinates, s: 20.0);
  final modernU = linspace(0.0, 1.0, 10);
  final modernResult = modernSpline.call(modernU);

  print('Modern result (10 points):');
  for (int i = 0; i < modernResult[0].length; i++) {
    final x = modernResult[0][i];
    final y = modernResult[1][i];
    print('  Modern [$i]: (${x.toStringAsFixed(1)}, ${y.toStringAsFixed(1)})');
  }

  // ===== COMPARISON =====
  print('\n🔍 Comparing Results:');
  double maxDifferenceX = 0.0, maxDifferenceY = 0.0;
  for (int i = 0; i < 10; i++) {
    final diffX = (legacyResult[0][i] - modernResult[0][i]).abs();
    final diffY = (legacyResult[1][i] - modernResult[1][i]).abs();
    maxDifferenceX = maxDifferenceX > diffX ? maxDifferenceX : diffX;
    maxDifferenceY = maxDifferenceY > diffY ? maxDifferenceY : diffY;
  }
  print('Maximum X difference: ${maxDifferenceX.toStringAsFixed(3)}');
  print('Maximum Y difference: ${maxDifferenceY.toStringAsFixed(3)}');
  print(
    'Interfaces are ${maxDifferenceX < 0.1 && maxDifferenceY < 0.1 ? "✅ equivalent" : "❌ different"}',
  );

  print('\n' + '=' * 60 + '\n');

  // Demo: Modern BSpline Properties
  print('🔧 Modern BSpline Object Properties');
  print('Exploring the BSpline object properties and capabilities');

  final (bspline, params) = makeSplprep(coordinates, k: 3);

  print('BSpline properties:');
  print('  - Dimensions: ${bspline.dimensions}');
  print('  - Degree (k): ${bspline.k}');
  print('  - Number of knots: ${bspline.t.length}');
  print(
    '  - Knot vector: ${bspline.t.map((x) => x.toStringAsFixed(2)).join(", ")}',
  );
  print(
    '  - Coefficient matrix shape: ${bspline.c.length} x ${bspline.c[0].length}',
  );

  print('\nParameter values:');
  for (int i = 0; i < params.length; i++) {
    print('  u[$i] = ${params[i].toStringAsFixed(3)}');
  }

  print('\n' + '=' * 60 + '\n');

  // Demo: Performance comparison
  print('⚡ Performance & Use Case Demo');
  print('Simulating high-frequency object tracking (like surveillance)');

  // Simulate a longer trajectory with more points
  final longX = List.generate(20, (i) => 50.0 + i * 5.0 + (i % 3) * 2.0);
  final longY = List.generate(20, (i) => 300.0 - i * 8.0 + (i % 4) * 3.0);
  final longCoords = [longX, longY];

  print('Long trajectory with ${longX.length} detection points');
  print(
    'First 5 points: ${longX.take(5).map((x) => x.toStringAsFixed(0)).join(", ")}',
  );
  print(
    'Last 5 points: ${longX.skip(15).map((x) => x.toStringAsFixed(0)).join(", ")}',
  );

  final stopwatch = Stopwatch()..start();

  // Process with modern interface
  final (longSpline, _) = makeSplprep(longCoords, s: 30.0);
  final denseU = linspace(
    0.0,
    1.0,
    100,
  ); // Dense sampling for smooth visualization
  final smoothTrajectory = longSpline.call(denseU);

  stopwatch.stop();

  print('\nProcessing results:');
  print('  - Input points: ${longX.length}');
  print('  - Output smooth points: ${smoothTrajectory[0].length}');
  print('  - Processing time: ${stopwatch.elapsedMicroseconds} microseconds');
  print(
    '  - Smoothness factor: 100 points generated from ${longX.length} detections',
  );

  // Verify smoothness
  double maxStep = 0.0;
  for (int i = 1; i < smoothTrajectory[0].length; i++) {
    final stepX = (smoothTrajectory[0][i] - smoothTrajectory[0][i - 1]).abs();
    final stepY = (smoothTrajectory[1][i] - smoothTrajectory[1][i - 1]).abs();
    final stepSize = stepX + stepY;
    maxStep = maxStep > stepSize ? maxStep : stepSize;
  }
  print(
    '  - Maximum step size: ${maxStep.toStringAsFixed(2)} (lower = smoother)',
  );

  print('\n' + '=' * 60 + '\n');

  // Demo: Real-world supervision usage pattern
  print('🎯 Real-World Usage Pattern (Supervision Style)');
  print('Demonstrating exact usage pattern from supervision library');

  // This exactly mimics how supervision uses splines in TraceAnnotator
  final detectionCenters = [
    [
      640.0,
      650.0,
      660.0,
      670.0,
      680.0,
      690.0,
    ], // x coordinates of object center
    [
      360.0,
      350.0,
      340.0,
      330.0,
      320.0,
      310.0,
    ], // y coordinates of object center
  ];

  print('Object detection centers over 6 frames:');
  for (int i = 0; i < detectionCenters[0].length; i++) {
    print(
      '  Frame $i: object at (${detectionCenters[0][i]}, ${detectionCenters[1][i]})',
    );
  }

  // Check supervision's condition: len(xy) > 3 and self.smooth
  final smooth = true;
  if (detectionCenters[0].length > 3 && smooth) {
    print('\n✅ Applying smoothing (supervision condition met)');

    // Exact supervision pattern:
    final x = detectionCenters[0];
    final y = detectionCenters[1];
    final (tck, u) = splprep([x, y], s: 20);
    final xNew = splev(linspace(0, 1, 100), tck)[0];
    final yNew = splev(linspace(0, 1, 100), tck)[1];

    // Convert to integer coordinates for OpenCV drawing
    final splinePoints = <List<int>>[];
    for (int i = 0; i < xNew.length; i++) {
      splinePoints.add([xNew[i].round(), yNew[i].round()]);
    }

    print(
      'Generated ${splinePoints.length} integer coordinate points for cv2.polylines()',
    );
    print('Ready for OpenCV drawing:');
    print('  - First point: ${splinePoints.first}');
    print('  - Last point: ${splinePoints.last}');
    print('  - Sample points: ${splinePoints.take(5).toList()}');

    // Verify the trajectory is reasonable
    final startDistance = ((splinePoints.first[0] - x.first).abs() +
        (splinePoints.first[1] - y.first).abs());
    final endDistance = ((splinePoints.last[0] - x.last).abs() +
        (splinePoints.last[1] - y.last).abs());

    print(
      '  - Start point accuracy: ${startDistance.toStringAsFixed(1)} pixels from original',
    );
    print(
      '  - End point accuracy: ${endDistance.toStringAsFixed(1)} pixels from original',
    );

    if (startDistance < 5 && endDistance < 5) {
      print('  ✅ Trajectory endpoints are accurate');
    } else {
      print('  ⚠️  Check trajectory accuracy');
    }
  } else {
    print('❌ Not enough points for smoothing (need > 3)');
  }

  print('\n🎉 Enhanced demo completed successfully!');
  print('Both legacy and modern interfaces are ready for production use.');
  print('The implementation is compatible with supervision library patterns.');
}
