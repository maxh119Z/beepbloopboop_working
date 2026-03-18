import SwiftUI

struct FormView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var showInfo = true  // your RN code starts visible=true

    var body: some View {
        ZStack {
            Color(hex: "#ffffff").ignoresSafeArea()

            VStack(spacing: 60) {
                // Top row buttons
                HStack {
                    IconButton(systemName: "chevron.left") {
                        dismiss() // goes back
                    }

                    Spacer()

                    IconButton(systemName: "info.circle") {
                        showInfo = true
                    }
                }
                .padding(.horizontal, 20)
                .padding(.top, 10)

                // Main section
                VStack(spacing: 12) {
                    NavigationLink {
                        CameraPoseView()
                    } label: {
                        HStack(spacing: 10) {
                            Text("Start Form Detection")
                                .font(.system(size: 22, weight: .heavy))

                            Image(systemName: "camera")
                                .font(.system(size: 22, weight: .bold))
                        }
                        .foregroundColor(Color(hex: "#2e1315"))
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 14)
                        .background(Color(hex: "#f7b5b8"))
                        .cornerRadius(10)
                        .shadow(color: .black.opacity(0.25), radius: 7, x: 0, y: 4)
                    }
                    .buttonStyle(.plain)


                    Text("Wifi Connection is Required!")
                        .font(.system(size: 16, weight: .semibold))
                        .italic()
                        .foregroundColor(Color(hex: "#1A202C"))

                    // placeholder image
                    Image("placeholder")
                        .resizable()
                        .scaledToFit()
                        .frame(width: 220)
                        .frame(maxWidth: .infinity)
                }
                .padding(.horizontal, 26)

                Spacer()
            }
        }
        .sheet(isPresented: $showInfo) {
            InfoModalView(showInfo: $showInfo)
                .presentationDetents([.fraction(0.55)])
        }
    }
}

private struct IconButton: View {
    let systemName: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: systemName)
                .font(.system(size: 22, weight: .bold))
                .foregroundColor(.black)
                .frame(width: 70, height: 60)
                .background(Color(hex: "#9fc9ae"))
                .cornerRadius(12)
                .shadow(color: .black.opacity(0.25), radius: 7, x: 0, y: 4)
        }
    }
}

private struct InfoModalView: View {
    @Binding var showInfo: Bool

    var body: some View {
        ZStack {
            Color(hex: "#ffffff").ignoresSafeArea()

            VStack(spacing: 12) {
                HStack {
                    Button {
                        showInfo = false
                    } label: {
                        Text("X")
                            .font(.system(size: 18, weight: .heavy))
                            .foregroundColor(.white)
                            .frame(width: 40, height: 40)
                            .background(Color(hex: "#f7b5b8"))
                            .cornerRadius(12)
                    }
                    Spacer()
                }
                .padding(.horizontal, 18)
                .padding(.top, 10)

                Text("""
Turn your device horizontally when your camera turns on.

Ensure your entire upper body is visible. Hold up your badminton form, and the software will map your joints. Green = Good, while Red = something is wrong!
""")
                .font(.system(size: 15, weight: .bold))
                .multilineTextAlignment(.center)
                .padding(.horizontal, 18)

                Image("formguy")
                    .resizable()
                    .scaledToFit()
                    .padding(.horizontal, 18)
                    .scaleEffect(x: -1, y: 1)

                Spacer(minLength: 10)
            }
        }
    }
}
