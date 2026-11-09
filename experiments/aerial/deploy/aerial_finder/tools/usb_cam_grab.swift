import AVFoundation
import CoreImage
import Foundation
import ImageIO

/// Capture the USB camera named "4000" and write the newest JPEG.
/// OpenCV indexes do not follow this name: the gray frame is the phone,
/// the face is the Mac camera.
final class Sink: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let ctx = CIContext()
    let url = URL(fileURLWithPath: "/tmp/sjcam_usb.jpg")
    let tmp = URL(fileURLWithPath: "/tmp/sjcam_usb.jpg.tmp")

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let image = CIImage(cvPixelBuffer: pb)
        guard let space = CGColorSpace(name: CGColorSpace.sRGB) else { return }
        do {
            try ctx.writeJPEGRepresentation(of: image, to: tmp, colorSpace: space, options: [:])
            if FileManager.default.fileExists(atPath: url.path) {
                _ = try FileManager.default.replaceItemAt(url, withItemAt: tmp)
            } else {
                try FileManager.default.moveItem(at: tmp, to: url)
            }
        } catch {
            fputs("jpeg \(error)\n", stderr)
        }
    }
}

let devs = AVCaptureDevice.devices(for: .video)
guard let dev = devs.first(where: { $0.localizedName == "4000" }) else {
    fputs("SJCAM 4000 is not on this Mac\n", stderr)
    for d in devs {
        fputs("dev \(d.localizedName) \(d.uniqueID)\n", stderr)
    }
    exit(1)
}
fputs("using \(dev.localizedName) \(dev.uniqueID)\n", stderr)

let session = AVCaptureSession()
session.beginConfiguration()
if session.canSetSessionPreset(.hd1280x720) {
    session.sessionPreset = .hd1280x720
}
do {
    let input = try AVCaptureDeviceInput(device: dev)
    if session.canAddInput(input) { session.addInput(input) }
} catch {
    fputs("input \(error)\n", stderr)
    exit(1)
}
let output = AVCaptureVideoDataOutput()
output.alwaysDiscardsLateVideoFrames = true
output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)]
let sink = Sink()
output.setSampleBufferDelegate(sink, queue: DispatchQueue(label: "usbcam"))
if session.canAddOutput(output) { session.addOutput(output) }
session.commitConfiguration()
session.startRunning()
dispatchMain()
