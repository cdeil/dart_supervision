import 'dart:io';
import 'package:test/test.dart';
import 'package:dart_supervision/dart_supervision.dart' as sv;

void main() {
  group('VideoAssets', () {
    test('should have correct filenames', () {
      expect(sv.VideoAssets.vehicles.filename, equals('vehicles.mp4'));
      expect(
        sv.VideoAssets.milkBottlingPlant.filename,
        equals('milk-bottling-plant.mp4'),
      );
      expect(sv.VideoAssets.vehicles2.filename, equals('vehicles-2.mp4'));
      expect(sv.VideoAssets.groceryStore.filename, equals('grocery-store.mp4'));
      expect(sv.VideoAssets.subway.filename, equals('subway.mp4'));
      expect(sv.VideoAssets.marketSquare.filename, equals('market-square.mp4'));
      expect(sv.VideoAssets.peopleWalking.filename, equals('people-walking.mp4'));
      expect(sv.VideoAssets.beach.filename, equals('beach-1.mp4'));
      expect(sv.VideoAssets.basketball.filename, equals('basketball-1.mp4'));
      expect(sv.VideoAssets.skiing.filename, equals('skiing.mp4'));
    });

    test('should generate correct URLs', () {
      expect(
        sv.VideoAssets.vehicles.url,
        equals(
          'https://media.roboflow.com/supervision/video-examples/vehicles.mp4',
        ),
      );
      expect(
        sv.VideoAssets.skiing.url,
        equals(
          'https://media.roboflow.com/supervision/video-examples/skiing.mp4',
        ),
      );
    });

    test('allFilenames should return all video filenames', () {
      final filenames = sv.VideoAssets.allFilenames;
      expect(filenames.length, equals(10));
      expect(filenames, contains('vehicles.mp4'));
      expect(filenames, contains('skiing.mp4'));
    });
  });

  group('sv.videoAssetsMetadata', () {
    test('should contain metadata for all video assets', () {
      expect(sv.videoAssetsMetadata.length, equals(10));

      final vehiclesMetadata = sv.videoAssetsMetadata['vehicles.mp4'];
      expect(vehiclesMetadata, isNotNull);
      expect(
        vehiclesMetadata!.url,
        equals(
          'https://media.roboflow.com/supervision/video-examples/vehicles.mp4',
        ),
      );
      expect(
        vehiclesMetadata.md5Hash,
        equals('8155ff4e4de08cfa25f39de96483f918'),
      );
    });
  });

  group('Asset downloader', () {
    late Directory tempDir;

    setUp(() async {
      tempDir = await Directory.systemTemp.createTemp('assets_test_');
    });

    tearDown(() async {
      if (await tempDir.exists()) {
        await tempDir.delete(recursive: true);
      }
    });

    test('should validate MD5 hash correctly', () async {
      // Create a test file with known content
      final testFile = File('${tempDir.path}/test.txt');
      await testFile.writeAsString('Hello, World!');

      // MD5 of "Hello, World!" is 65a8e27d8879283831b664bd8b7f0ad4
      final isValid = await sv.isMd5HashMatching(
        testFile.path,
        '65a8e27d8879283831b664bd8b7f0ad4',
      );
      expect(isValid, isTrue);

      // Test with incorrect hash
      final isInvalid = await sv.isMd5HashMatching(
        testFile.path,
        'incorrect_hash',
      );
      expect(isInvalid, isFalse);
    });

    test('should handle non-existent files', () async {
      final nonExistentFile = '${tempDir.path}/non_existent.txt';
      final isValid = await sv.isMd5HashMatching(nonExistentFile, 'any_hash');
      expect(isValid, isFalse);
    });

    test('should reject invalid asset names', () async {
      expect(
        () => sv.downloadAssets('invalid_asset.mp4', outputDir: tempDir.path),
        throwsA(isA<ArgumentError>()),
      );
    });

    test('should accept VideoAssets enum values', () async {
      // This test would actually download if we had a mock server
      // For now, we just test that it doesn't throw immediately
      expect(
        () => sv.downloadAssets(sv.VideoAssets.vehicles, outputDir: tempDir.path),
        returnsNormally,
      );
    });

    test('should accept string asset names', () async {
      // This test would actually download if we had a mock server
      // For now, we just test that it doesn't throw immediately
      expect(
        () => sv.downloadAssets('vehicles.mp4', outputDir: tempDir.path),
        returnsNormally,
      );
    });
  });
}
