import SwiftUI
import WebKit

/// 良子语音助手的服务端地址（跑在你 Mac 上的 orchestrator）。
/// 换网络/换机器时改这里：https://<Mac 局域网 IP>:9443
/// 证书由仓库 scripts/make_lan_tls.sh 生成，iPhone 需先信任根证书（见脚本输出指引）。
private let serverURL = URL(string: "https://192.168.2.20:9443")!

@main
struct LiangziVoiceApp: App {
    var body: some Scene {
        WindowGroup {
            WebPageView(url: serverURL)
                .ignoresSafeArea()
                .onAppear {
                    // 通话期间不自动锁屏
                    UIApplication.shared.isIdleTimerDisabled = true
                }
        }
    }
}

/// 全屏 WKWebView 壳：页面/星空/通话逻辑全在服务端 web/ 里，
/// 壳只负责两件事——给麦克风权限、提供一个 https 安全上下文。
struct WebPageView: UIViewRepresentable {
    let url: URL

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeUIView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.allowsInlineMediaPlayback = true
        // 音频自动播放交给页面按钮手势处理（iOS 惯例）
        config.mediaTypesRequiringUserActionForPlayback = []
        let webView = WKWebView(frame: .zero, configuration: config)
        webView.uiDelegate = context.coordinator
        webView.scrollView.bounces = false
        webView.load(URLRequest(url: url))
        return webView
    }

    func updateUIView(_ uiView: WKWebView, context: Context) {}

    final class Coordinator: NSObject, WKUIDelegate {
        /// iOS 15+：getUserMedia 在 WKWebView 里会经这个回调要权限。
        /// 直接 grant——系统级麦克风弹窗仍会出现一次（首次使用），之后记住。
        func webView(_ webView: WKWebView,
                     requestMediaCapturePermissionFor origin: WKSecurityOrigin,
                     initiatedByFrame frame: WKFrameInfo,
                     type: WKMediaCaptureType,
                     decisionHandler: @escaping (WKPermissionDecision) -> Void) {
            decisionHandler(.grant)
        }
    }
}
