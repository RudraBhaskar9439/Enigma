export default function Footer() {
  return (
    <footer className="footer">
      <div className="footer-inner">
        <div>
          <div className="brand" style={{ marginBottom: 12 }}>
            <div className="brand-logo" />
            <span>Vishwamitra</span>
          </div>
          <p className="footer-brand-text">
            An agentic grievance resolution platform — built with empathic LLMs to make
            every voice heard.
          </p>
        </div>
        <div>
          <h4>Product</h4>
          <ul>
            <li><a href="#features">Features</a></li>
            <li><a href="#how">How it works</a></li>
            <li><a href="#modes">Get started</a></li>
          </ul>
        </div>
        <div>
          <h4>Company</h4>
          <ul>
            <li><a href="#">About</a></li>
            <li><a href="#">Contact</a></li>
            <li><a href="#">Privacy</a></li>
          </ul>
        </div>
        <div>
          <h4>Resources</h4>
          <ul>
            <li><a href="#">Docs</a></li>
            <li><a href="#">Support</a></li>
          </ul>
        </div>
      </div>
      <div className="footer-bottom">
        <span>© {new Date().getFullYear()} Vishwamitra. All rights reserved.</span>
        <span>Made with care · powered by LLMs</span>
      </div>
    </footer>
  )
}
