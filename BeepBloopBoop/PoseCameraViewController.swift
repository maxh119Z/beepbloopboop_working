//
//  PoseCameraViewController.swift
//  bbbabababdfdf
//
//  Created by Max Zhang on 2/4/26.
//

import UIKit
import AVFoundation
import Vision

final class PoseCameraViewController: UIViewController {

    // MARK: - Timing
    private let visionInterval: CFTimeInterval = 1.0 / 30.0
    private var lastVisionTime = CFAbsoluteTimeGetCurrent()

    // MARK: - Smoothing (normalized space)
    private let holdSeconds: CFTimeInterval = 0.35
    private let minConf: Float = 0.10

    private var smoothedNorm: [VNHumanBodyPoseObservation.JointName: CGPoint] = [:]
    private var lastGoodTime: [VNHumanBodyPoseObservation.JointName: CFTimeInterval] = [:]

    private let alphaHigh: CGFloat = 0.35
    private let alphaLow: CGFloat  = 0.15

    // MARK: - Angle smoothing
    private var smoothC: CGFloat?
    private var smoothD: CGFloat?
    private var smoothNdC: CGFloat?
    private var smoothNdD: CGFloat?

    // MARK: - Hysteresis state
    private var elbowGood = true
    private var shoulderGood = true
    private var shoulder2Good = true
    private var elbow2Good = true

    private var elbow2BadSince: CFTimeInterval? = nil
    private let elbow2Dwell: CFTimeInterval = 0.12

    private var elbow2GoodSince: CFTimeInterval? = nil
    private let elbow2RecoverDwell: CFTimeInterval = 0.10

    // MARK: - UI state
    private var isLeftyMode = false

    // MARK: - Layers
    private let overlayLayer = CAShapeLayer()
    private let elbowMarker = CAShapeLayer()
    private let shoulderMarker = CAShapeLayer()
    private let elbow2Marker = CAShapeLayer()
    private let shoulder2Marker = CAShapeLayer()

    private let jointRadius: CGFloat = 5.0
    private let markerRadius: CGFloat = 16.0

    // MARK: - Camera
    private let captureSession = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()

    private let videoQueue = DispatchQueue(label: "camera.video.queue", qos: .userInitiated)
    private let captureQueue = DispatchQueue(label: "camera.session.queue", qos: .userInitiated)

    private var previewLayer: AVCaptureVideoPreviewLayer!

    // MARK: - Vision
    private let sequenceHandler = VNSequenceRequestHandler()
    private let poseRequest = VNDetectHumanBodyPoseRequest()

    // MARK: - Lifecycle
    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black

        setupPreview()
        setupOverlay()
        setupCamera()
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer.frame = view.bounds
        overlayLayer.frame = view.bounds

        elbowMarker.frame = view.bounds
        shoulderMarker.frame = view.bounds
        elbow2Marker.frame = view.bounds
        shoulder2Marker.frame = view.bounds
    }

    // MARK: - Setup
    private func setupPreview() {
        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.videoGravity = .resizeAspectFill

        if let c = previewLayer.connection, c.isVideoMirroringSupported {
            c.videoOrientation = .portrait
            c.isVideoMirrored = true
        }

        view.layer.addSublayer(previewLayer)
    }

    private func setupOverlay() {
        // Skeleton/joints
        overlayLayer.frame = view.bounds
        overlayLayer.strokeColor = UIColor.white.cgColor
        overlayLayer.fillColor = UIColor.clear.cgColor
        overlayLayer.lineWidth = 3.0
        overlayLayer.lineJoin = .round
        overlayLayer.lineCap = .round

        // Disable implicit animations (super important for “clean” drawing)
        overlayLayer.actions = ["path": NSNull()]

        view.layer.addSublayer(overlayLayer)

        func setupMarker(_ layer: CAShapeLayer) {
            layer.frame = view.bounds
            layer.strokeColor = UIColor.clear.cgColor
            layer.fillColor = UIColor.red.cgColor
            layer.lineWidth = 3
            layer.actions = [
                "path": NSNull(),
                "fillColor": NSNull()
            ]
            view.layer.addSublayer(layer)
        }

        setupMarker(elbowMarker)
        setupMarker(shoulderMarker)
        setupMarker(elbow2Marker)
        setupMarker(shoulder2Marker)

        overlayLayer.zPosition = 10
        elbowMarker.zPosition = 11
        shoulderMarker.zPosition = 11
        elbow2Marker.zPosition = 11
        shoulder2Marker.zPosition = 11
    }

    private func setupCamera() {
        captureSession.beginConfiguration()
        captureSession.sessionPreset = .high

        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front),
              let input = try? AVCaptureDeviceInput(device: device),
              captureSession.canAddInput(input) else {
            print("❌ Failed to create camera input.")
            captureSession.commitConfiguration()
            return
        }
        captureSession.addInput(input)

        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: videoQueue)

        guard captureSession.canAddOutput(videoOutput) else {
            print("❌ Failed to add video output.")
            captureSession.commitConfiguration()
            return
        }
        captureSession.addOutput(videoOutput)

        if let connection = videoOutput.connection(with: .video) {
            connection.videoOrientation = .portrait
            if connection.isVideoMirroringSupported {
                connection.isVideoMirrored = true
            }
        }

        captureSession.commitConfiguration()

        // Start on background queue (fixes your warning)
        captureQueue.async { [weak self] in
            self?.captureSession.startRunning()
        }
    }

    // MARK: - Processing
    private func processPose(pixelBuffer: CVPixelBuffer) {
        let now = CFAbsoluteTimeGetCurrent()
        guard now - lastVisionTime >= visionInterval else { return }
        lastVisionTime = now

        do {
            // ✅ front camera portrait mirrored
            let orientation: CGImagePropertyOrientation = .rightMirrored
            try sequenceHandler.perform([poseRequest], on: pixelBuffer, orientation: orientation)

            guard let observation = poseRequest.results?.first else {
                DispatchQueue.main.async {
                    self.overlayLayer.path = nil
                    self.elbowMarker.path = nil
                    self.shoulderMarker.path = nil
                    self.elbow2Marker.path = nil
                    self.shoulder2Marker.path = nil
                }
                return
            }

            let points = try observation.recognizedPoints(.all)
            let nowT = CFAbsoluteTimeGetCurrent()

            // Smoothed SCREEN point
            func PS(_ name: VNHumanBodyPoseObservation.JointName) -> CGPoint? {
                guard let n = smoothedNormalizedPoint(joint: name, point: points[name], now: nowT) else { return nil }
                return toScreen(n)
            }

            var psCache: [VNHumanBodyPoseObservation.JointName: CGPoint] = [:]
            func PSCached(_ name: VNHumanBodyPoseObservation.JointName) -> CGPoint? {
                if let v = psCache[name] { return v }
                let v = PS(name)
                if let v { psCache[name] = v }
                return v
            }

            // Draw skeleton + joints
            let (jointsPath, skeletonPath) = makePathsSmoothed(PS: PSCached)

            DispatchQueue.main.async {
                let combined = UIBezierPath()
                combined.append(skeletonPath)
                combined.append(jointsPath)
                self.overlayLayer.path = combined.cgPath
            }

            // ---- Markers / angle logic needs these 6 joints in screen space ----
            let rsName: VNHumanBodyPoseObservation.JointName = isLeftyMode ? .leftShoulder : .rightShoulder
            let lsName: VNHumanBodyPoseObservation.JointName = isLeftyMode ? .rightShoulder : .leftShoulder
            let reName: VNHumanBodyPoseObservation.JointName = isLeftyMode ? .leftElbow : .rightElbow
            let rwName: VNHumanBodyPoseObservation.JointName = isLeftyMode ? .leftWrist : .rightWrist
            let leName: VNHumanBodyPoseObservation.JointName = isLeftyMode ? .rightElbow : .leftElbow
            let lwName: VNHumanBodyPoseObservation.JointName = isLeftyMode ? .rightWrist : .leftWrist

            guard
                let rsS = PSCached(rsName),
                let lsS = PSCached(lsName),
                let reS = PSCached(reName),
                let rwS = PSCached(rwName),
                let leS = PSCached(leName),
                let lwS = PSCached(lwName)
            else {
                DispatchQueue.main.async {
                    self.elbowMarker.path = nil
                    self.shoulderMarker.path = nil
                    self.elbow2Marker.path = nil
                    self.shoulder2Marker.path = nil
                }
                return
            }

            // Vectors
            let lrshoulder = Vec(x: rsS.x - lsS.x, y: rsS.y - lsS.y)
            let ellshoulder = Vec(x: lsS.x - leS.x, y: lsS.y - leS.y)

            var ndD = angleDegrees(lrshoulder, ellshoulder)
            let ns1 = slope(lsS, rsS)
            let ns2 = slope(lsS, leS)
            if ns2 >= ns1 { ndD *= -1 }

            let lewrist = Vec(x: lwS.x - leS.x, y: lwS.y - leS.y)
            let ndC = angleDegrees(lewrist, ellshoulder)

            let elshoulder = Vec(x: rsS.x - reS.x, y: rsS.y - reS.y)
            let rewrist = Vec(x: rwS.x - reS.x, y: rwS.y - reS.y)
            let C = angleDegrees(rewrist, elshoulder)

            let rlshoulder = Vec(x: lsS.x - rsS.x, y: lsS.y - rsS.y)
            var D = angleDegrees(rlshoulder, elshoulder)
            let s1 = slope(rsS, lsS)
            let s2 = slope(rsS, reS)
            if s2 < s1 { D *= -1 }

            // Smooth angles
            let aAngle: CGFloat = 0.25
            smoothC   = ema(smoothC,   C,   aAngle)
            smoothD   = ema(smoothD,   D,   aAngle)
            smoothNdC = ema(smoothNdC, ndC, aAngle)
            smoothNdD = ema(smoothNdD, ndD, aAngle)

            let Cuse   = smoothC ?? C
            let Duse   = smoothD ?? D
            let ndCuse = smoothNdC ?? ndC
            let ndDuse = smoothNdD ?? ndD

            let goodGreen = UIColor(red: 0x0F/255.0, green: 0xFF/255.0, blue: 0x50/255.0, alpha: 1)
            let m: CGFloat = 4.0

            elbowGood = hysteresis(elbowGood, value: Cuse, goodMin: 35, goodMax: 98.1, margin: m)
            let elbowColor: UIColor = elbowGood ? goodGreen : .red

            if isLeftyMode {
                shoulderGood  = hysteresis(shoulderGood,  value: Duse,   goodMin: -10,  goodMax: 30,   margin: m)
                shoulder2Good = hysteresis(shoulder2Good, value: ndDuse,  goodMin: -31.3, goodMax: 16.7, margin: m)
            } else {
                shoulderGood  = hysteresis(shoulderGood,  value: Duse,   goodMin: -31.3, goodMax: 16.7, margin: m)
                shoulder2Good = hysteresis(shoulder2Good, value: ndDuse,  goodMin: -10,  goodMax: 30,   margin: m)
            }

            let shoulderColor: UIColor  = shoulderGood ? goodGreen : .red
            let shoulder2Color: UIColor = shoulder2Good ? goodGreen : .red

            let badElbow2 = (ndCuse < 90.0 || ndCuse > 180.0 || (ndCuse + Cuse) < 120.0 || (ndCuse + Cuse) > 240.0)

            if badElbow2 {
                elbow2GoodSince = nil
                if elbow2BadSince == nil { elbow2BadSince = nowT }
                if nowT - (elbow2BadSince ?? nowT) >= elbow2Dwell {
                    elbow2Good = false
                }
            } else {
                elbow2BadSince = nil
                if elbow2GoodSince == nil { elbow2GoodSince = nowT }
                if nowT - (elbow2GoodSince ?? nowT) >= elbow2RecoverDwell {
                    elbow2Good = true
                }
            }

            let elbow2Color: UIColor = elbow2Good ? goodGreen : .red

            // Draw circles
            DispatchQueue.main.async {
                self.elbowMarker.fillColor = elbowColor.cgColor
                self.elbowMarker.path = self.circlePath(center: reS, radius: self.markerRadius)

                self.shoulderMarker.fillColor = shoulderColor.cgColor
                self.shoulderMarker.path = self.circlePath(center: rsS, radius: self.markerRadius)

                self.shoulder2Marker.fillColor = shoulder2Color.cgColor
                self.shoulder2Marker.path = self.circlePath(center: lsS, radius: self.markerRadius)

                self.elbow2Marker.fillColor = elbow2Color.cgColor
                self.elbow2Marker.path = self.circlePath(center: leS, radius: self.markerRadius)
            }

        } catch {
            print("Vision error:", error)
        }
    }

    // MARK: - Coordinate conversion
    private func toScreen(_ p: CGPoint) -> CGPoint {
        // p is normalized (0..1) in Vision/capture-device space.
        // If the preview is mirrored (selfie camera style), the overlay must mirror too.
        var cp = p
        if previewLayer.connection?.isVideoMirrored == true {
            cp.x = 1.0 - cp.x
        }
        return previewLayer.layerPointConverted(fromCaptureDevicePoint: cp)
    }


    // MARK: - Smoothing
    private func smoothedNormalizedPoint(
        joint: VNHumanBodyPoseObservation.JointName,
        point: VNRecognizedPoint?,
        now: CFTimeInterval
    ) -> CGPoint? {

        if let p = point, p.confidence >= minConf {
            let raw = CGPoint(x: p.x, y: p.y)
            let prev = smoothedNorm[joint] ?? raw
            let a: CGFloat = (p.confidence >= 0.6) ? alphaHigh : alphaLow
            let out = lerp(prev, raw, a)

            smoothedNorm[joint] = out
            lastGoodTime[joint] = now
            return out
        }

        if let t = lastGoodTime[joint],
           now - t <= holdSeconds,
           let held = smoothedNorm[joint] {
            return held
        }

        return nil
    }

    private func lerp(_ a: CGPoint, _ b: CGPoint, _ t: CGFloat) -> CGPoint {
        CGPoint(x: a.x + (b.x - a.x) * t,
                y: a.y + (b.y - a.y) * t)
    }

    // MARK: - Paths
    private func makePathsSmoothed(PS: (VNHumanBodyPoseObservation.JointName) -> CGPoint?)
    -> (UIBezierPath, UIBezierPath) {

        let jointsPath = UIBezierPath()
        let skeletonPath = UIBezierPath()

        func addDot(_ name: VNHumanBodyPoseObservation.JointName) {
            guard let p = PS(name) else { return }
            jointsPath.append(UIBezierPath(arcCenter: p, radius: jointRadius, startAngle: 0, endAngle: .pi * 2, clockwise: true))
        }

        func addLine(_ a: VNHumanBodyPoseObservation.JointName, _ b: VNHumanBodyPoseObservation.JointName) {
            guard let pa = PS(a), let pb = PS(b) else { return }
            skeletonPath.move(to: pa)
            skeletonPath.addLine(to: pb)
        }

        let dotJoints: [VNHumanBodyPoseObservation.JointName] = [
            .nose, .neck, .root,
            .leftShoulder, .leftElbow, .leftWrist,
            .rightShoulder, .rightElbow, .rightWrist,
            .leftHip, .leftKnee, .leftAnkle,
            .rightHip, .rightKnee, .rightAnkle
        ]
        for name in dotJoints { addDot(name) }

        addLine(.neck, .root)

        addLine(.neck, .leftShoulder);  addLine(.leftShoulder, .leftElbow);  addLine(.leftElbow, .leftWrist)
        addLine(.neck, .rightShoulder); addLine(.rightShoulder, .rightElbow); addLine(.rightElbow, .rightWrist)

        addLine(.root, .leftHip);  addLine(.leftHip, .leftKnee);  addLine(.leftKnee, .leftAnkle)
        addLine(.root, .rightHip); addLine(.rightHip, .rightKnee); addLine(.rightKnee, .rightAnkle)

        addLine(.leftShoulder, .rightShoulder)
        addLine(.leftHip, .rightHip)

        return (jointsPath, skeletonPath)
    }

    // MARK: - Geometry
    private struct Vec { var x: CGFloat; var y: CGFloat }

    private func dot(_ a: Vec, _ b: Vec) -> CGFloat { a.x * b.x + a.y * b.y }
    private func norm(_ v: Vec) -> CGFloat { sqrt(v.x * v.x + v.y * v.y) }

    private func angleDegrees(_ a: Vec, _ b: Vec) -> CGFloat {
        let denom = max(norm(a) * norm(b), 1e-6)
        var c = dot(a, b) / denom
        c = min(1.0, max(-1.0, c))
        return acos(c) * 180.0 / .pi
    }

    private func slope(_ a: CGPoint, _ b: CGPoint) -> CGFloat {
        let dx = (b.x - a.x)
        if abs(dx) < 1e-6 { return .infinity }
        return (b.y - a.y) / dx
    }

    private func circlePath(center: CGPoint, radius: CGFloat) -> CGPath {
        UIBezierPath(arcCenter: center, radius: radius, startAngle: 0, endAngle: .pi * 2, clockwise: true).cgPath
    }

    // MARK: - Filters
    private func ema(_ prev: CGFloat?, _ x: CGFloat, _ a: CGFloat) -> CGFloat {
        guard let prev else { return x }
        return prev + (x - prev) * a
    }

    private func hysteresis(_ currentGood: Bool, value: CGFloat, goodMin: CGFloat, goodMax: CGFloat, margin: CGFloat) -> Bool {
        if currentGood {
            return value >= (goodMin - margin) && value <= (goodMax + margin)
        } else {
            return value >= (goodMin + margin) && value <= (goodMax - margin)
        }
    }
}

extension PoseCameraViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        processPose(pixelBuffer: pixelBuffer)
    }
}
