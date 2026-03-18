import SwiftUI
import UIKit

struct CameraPoseView: View {
    var body: some View {
        PoseCameraViewControllerRepresentable()
            .ignoresSafeArea() // camera full-screen
            .navigationBarTitleDisplayMode(.inline)
    }
}

private struct PoseCameraViewControllerRepresentable: UIViewControllerRepresentable {
    func makeUIViewController(context: Context) -> PoseCameraViewController {
        PoseCameraViewController()
    }

    func updateUIViewController(_ uiViewController: PoseCameraViewController, context: Context) {
    }
}
