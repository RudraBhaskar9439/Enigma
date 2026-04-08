export default function Navbar({ onHome, onLaunch, onNavigate }) {
  function go(id) {
    if (onNavigate) onNavigate()
    requestAnimationFrame(() => {
      document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    })
  }
  return (
    <header className="nav">
      <div className="nav-inner">
        <div className="brand" onClick={onHome} style={{ cursor: 'pointer' }}>
          <div className="brand-logo" />
          <span>Vishwamitra</span>
        </div>
        <nav className="nav-links">
          <button onClick={() => go('features')}>Features</button>
          <button onClick={() => go('how')}>How it works</button>
          <button onClick={() => go('get-started')}>Get started</button>
        </nav>
        <button className="nav-cta" onClick={onLaunch}>Launch app</button>
      </div>
    </header>
  )
}
