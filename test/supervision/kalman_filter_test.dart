import 'package:test/test.dart';
import 'package:dart_supervision/dart_supervision.dart' as sv;

void main() {
  group('Kalman Filter', () {
    test('initializes track state correctly', () {
      final filter = sv.KalmanFilter();
      final measurement = sv.NDArray(
        [4],
        data: [10.0, 20.0, 1.5, 30.0],
      ); // x, y, aspect, height

      final result = filter.initiate(measurement);
      final mean = result.$1;
      final covariance = result.$2;

      expect(mean.shape, equals([8])); // [x, y, a, h, vx, vy, va, vh]
      expect(covariance.shape, equals([8, 8]));
      expect(mean[[0]], equals(10.0)); // x
      expect(mean[[1]], equals(20.0)); // y
      expect(mean[[4]], equals(0.0)); // vx initialized to 0
    });

    test('predicts next state', () {
      final filter = sv.KalmanFilter();
      final measurement = sv.NDArray([4], data: [10.0, 20.0, 1.0, 30.0]);

      final initResult = filter.initiate(measurement);
      final mean = initResult.$1.copy();
      final covariance = initResult.$2.copy();

      // Set some velocity
      mean[[4]] = 1.0; // vx = 1
      mean[[5]] = 2.0; // vy = 2

      final predResult = filter.predict(mean, covariance);
      final predMean = predResult.$1;
      final predCov = predResult.$2;

      expect(predMean.shape, equals([8]));
      expect(predCov.shape, equals([8, 8]));
      expect(predMean[[0]], equals(11.0)); // x += vx
      expect(predMean[[1]], equals(22.0)); // y += vy
    });

    test('updates state with measurement', () {
      final filter = sv.KalmanFilter();
      final measurement1 = sv.NDArray([4], data: [10.0, 20.0, 1.0, 30.0]);

      final initResult = filter.initiate(measurement1);
      final mean = initResult.$1;
      final covariance = initResult.$2;

      final measurement2 = sv.NDArray([4], data: [12.0, 22.0, 1.0, 30.0]);
      final updateResult = filter.update(mean, covariance, measurement2);
      final updatedMean = updateResult.$1;
      final updatedCov = updateResult.$2;

      expect(updatedMean.shape, equals([8]));
      expect(updatedCov.shape, equals([8, 8]));
      // Updated position should be influenced by new measurement
      expect(updatedMean[[0]], greaterThan(10.0));
      expect(updatedMean[[0]], lessThan(12.0));
    });
  });

  group('sv.STrack', () {
    test('initializes track correctly', () {
      final filter = sv.KalmanFilter();
      final tlwh = sv.NDArray(
        [4],
        data: [10.0, 20.0, 30.0, 40.0],
      ); // top, left, width, height

      final track = sv.STrack(tlwh, 0.9, filter);

      expect(track.trackId, equals(1));
      expect(track.score, equals(0.9));
      expect(track.state, equals(sv.TrackState.tentative));
      expect(track.timeSinceUpdate, equals(0));
      expect(track.hits, equals(1));
    });

    test('predicts track position', () {
      final filter = sv.KalmanFilter();
      final tlwh = sv.NDArray([4], data: [10.0, 20.0, 30.0, 40.0]);
      final track = sv.STrack(tlwh, 0.9, filter);

      track.predict();

      expect(track.timeSinceUpdate, equals(1));
      final bbox = track.tlwh;
      expect(bbox.shape, equals([4]));
    });

    test('updates track with new detection', () {
      final filter = sv.KalmanFilter();
      final tlwh = sv.NDArray([4], data: [10.0, 20.0, 30.0, 40.0]);
      final track = sv.STrack(tlwh, 0.9, filter);

      final newTlwh = sv.NDArray([4], data: [12.0, 22.0, 30.0, 40.0]);
      track.update(newTlwh, 0.8);

      expect(track.timeSinceUpdate, equals(0));
      expect(track.hits, equals(2));
      expect(track.score, equals(0.8));
    });

    test('marks track as missed', () {
      final filter = sv.KalmanFilter();
      final tlwh = sv.NDArray([4], data: [10.0, 20.0, 30.0, 40.0]);
      final track = sv.STrack(tlwh, 0.9, filter);

      track.markMissed();
      expect(track.state, equals(sv.TrackState.deleted)); // Tentative -> deleted

      // Confirmed track should not be deleted immediately
      track.state = sv.TrackState.confirmed;
      track.timeSinceUpdate = 0;
      track.markMissed();
      expect(track.state, equals(sv.TrackState.confirmed));

      // But should be deleted after too many misses
      track.timeSinceUpdate = 35;
      track.markMissed();
      expect(track.state, equals(sv.TrackState.deleted));
    });

    test('transitions from tentative to confirmed', () {
      final filter = sv.KalmanFilter();
      final tlwh = sv.NDArray([4], data: [10.0, 20.0, 30.0, 40.0]);
      final track = sv.STrack(tlwh, 0.9, filter);

      expect(track.isTentative, isTrue);
      expect(track.isConfirmed, isFalse);

      // Need 3 hits to confirm
      track.update(tlwh, 0.9);
      expect(track.isTentative, isTrue);

      track.update(tlwh, 0.9);
      expect(track.isConfirmed, isTrue);
      expect(track.isTentative, isFalse);
    });

    test('gets xyxy format bbox', () {
      final filter = sv.KalmanFilter();
      final tlwh = sv.NDArray(
        [4],
        data: [10.0, 20.0, 30.0, 40.0],
      ); // x=10, y=20, w=30, h=40
      final track = sv.STrack(tlwh, 0.9, filter);

      final xyxy = track.xyxy;
      expect(xyxy.shape, equals([4]));
      expect(xyxy[[0]], equals(10.0)); // x1
      expect(xyxy[[1]], equals(20.0)); // y1
      expect(xyxy[[2]], equals(40.0)); // x2 = x1 + w
      expect(xyxy[[3]], equals(60.0)); // y2 = y1 + h
    });

    test('resets ID counter', () {
      sv.STrack.resetIdCounter(); // Reset before the test
      final filter = sv.KalmanFilter();
      final tlwh = sv.NDArray([4], data: [10.0, 20.0, 30.0, 40.0]);

      final track1 = sv.STrack(tlwh, 0.9, filter);
      final track2 = sv.STrack(tlwh, 0.9, filter);

      expect(track1.trackId, equals(1));
      expect(track2.trackId, equals(2));

      sv.STrack.resetIdCounter();
      final track3 = sv.STrack(tlwh, 0.9, filter);
      expect(track3.trackId, equals(1));
    });
  });
}
