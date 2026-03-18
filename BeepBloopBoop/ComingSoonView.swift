import SwiftUI

struct ComingSoonView: View {
    let title: String

    var body: some View {
        ZStack {
            Color(hex: "#fdf6ec").ignoresSafeArea()
            VStack(spacing: 16) {
                Text(title)
                    .font(.system(size: 28, weight: .bold))
                Text("Coming soon…")
                    .font(.system(size: 18))
                    .foregroundColor(.secondary)
            }
        }
        .navigationTitle("")
        .navigationBarTitleDisplayMode(.inline)
    }
}
