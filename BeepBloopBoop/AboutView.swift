import SwiftUI

struct AboutView: View {
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        ZStack {
            Color(hex: "#fdf6ec")
                .ignoresSafeArea()

            VStack(alignment: .leading) {

                // Header with Back Button
                HStack {
                    Button {
                        dismiss()
                    } label: {
                        Image(systemName: "chevron.left")
                            .font(.system(size: 22, weight: .bold))
                            .foregroundColor(Color(hex: "#111111"))
                            .frame(width: 70, height: 60)
                            .background(Color(hex: "#9fc9ae"))
                            .cornerRadius(12)
                            .shadow(color: .black.opacity(0.1), radius: 7, x: 0, y: 4)
                    }

                    Spacer()
                }
                .padding(.top, 10)
                .padding(.horizontal, 20)

                ScrollView {
                    VStack(spacing: 35) {
                        Text("App Creators")
                            .font(.system(size: 32, weight: .bold))
                            .foregroundColor(.black)
                            .padding(.top, 20)

                        CreatorCard(
                            imageName: "formguy",
                            name: "Max",
                            bio: "max is so cool. max is so cool. max is so cool. max is so cool. max is so cool."
                        )
                        CreatorCard(
                            imageName: "formguy",
                            name: "Angela",
                            bio: "angela is so cool. angela is so cool. angela is so cool. angela is so cool. angela is so cool."
                        )

                        
                    }
                    .padding(.horizontal, 20)
                    .padding(.bottom, 40)
                }
            }
        }
    }
}

private struct CreatorCard: View {
    let imageName: String
    let name: String
    let bio: String

    var body: some View {
        VStack(spacing: 15) {
            Image(imageName)
                .resizable()
                .scaledToFill()
                .frame(width: 100, height: 100)
                .clipShape(Circle())

            Text(name)
                .font(.system(size: 24, weight: .semibold))
                .foregroundColor(.black)

            Text(bio)
                .font(.system(size: 16))
                .foregroundColor(Color(hex: "#555555"))
                .multilineTextAlignment(.center)
                .lineSpacing(6)
        }
        .padding(25)
        .frame(maxWidth: .infinity)
        .background(Color(hex: "#f8f8f2"))
        .cornerRadius(15)
        .shadow(color: .black.opacity(0.1), radius: 7, x: 0, y: 4)
    }
}
