import SwiftUI

struct ContentView: View {
    var body: some View {
        NavigationStack {
            GeometryReader { geo in
                ZStack {
                    Image("background")
                        .resizable()
                        .ignoresSafeArea()
                    
                    VStack(spacing: -18) {
                        
                        Spacer()
                            .frame(height: geo.size.height * 0.325)
                        
                        VStack(spacing: 9) {
                            VStack(spacing: -5){
                                Text("BIRDIE")
                                    .font(.custom("Horizon", size: 55))
                                    .foregroundColor(Color(hex: "#2a4c09"))
                                    .multilineTextAlignment(.center)
                                    .shadow(color: .black.opacity(0.30), radius: 10, x: 0, y: 6)
                                    .shadow(color: .white.opacity(0.60), radius: 3, x: 0, y: -1)
                                
                                Rectangle()
                                    .fill(Color(hex: "#2a4c09"))
                                    .frame(width: 300, height: 4)
                                    .shadow(color: .black.opacity(0.25), radius: 6, x: 0, y: 4)
                                    .shadow(color: .white.opacity(0.55), radius: 2, x: 0, y: -1)
                            }
                            
                            
                            Text("Badminton Form Analysis")
                                .font(.custom("Etna", size: 30))
                                .foregroundColor(Color(hex: "#7ba63e"))
                            // Inner glow
                                .shadow(color: Color(hex: "#a0dd7f").opacity(0.7),
                                        radius: 4, x: 0, y: 0)
                            // Outer glow
                                .shadow(color: Color(hex: "#a0dd7f").opacity(0.45),
                                        radius: 10, x: 0, y: 0)
                            
                        }
                        //.padding(.top, 80)
                        //.padding(.bottom, 50)
                        Spacer()
                            .frame(height: geo.size.height * 0.08)
                        
                        VStack(spacing: 10) {
                            NavButton(title: "FORM CORRECTION") {
                                FormView()
                            }
                            
                            NavButton(title: "ABOUT US") {
                                AboutView()
                            }
                        }
                        .frame(width: geo.size.width * 0.78) // <- key: not full width
                        .frame(maxWidth: .infinity)
                        
                        Spacer()
                    }
                    .padding(.horizontal, 24)
                }
            }
        }
    }
}

/// Reusable button that matches your RN styling
private struct NavButton<Destination: View>: View {
    let title: String
    let destination: () -> Destination

    var body: some View {
            NavigationLink(destination: destination()) {
                Text(title)
                    .font(.custom("Etna", size: 26))
                    .foregroundColor(Color(hex: "#2a4c09"))         // dark green text
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 18)
                    .background(
                        Capsule()
                            .fill(Color(hex: "#BFE6A8"))            // light green pill
                    )
            }
            .buttonStyle(.plain)
    }
}
