export default function GetStarted({ onSimulator, onVoice }) {
  return (
    <section id="get-started" className="section">
      <div className="section-eyebrow">Get started</div>
      <h2 className="section-title">Pick how you want to talk to Vishwamitra</h2>
      <p className="section-lede">Two interfaces, one brain. Use the simulator for full control, or the voice mode for an empathic conversation.</p>
      <div className="get-grid">
        <div className="get-card">
          <div className="get-icon" aria-hidden>◧</div>
          <h3>Open the Simulator</h3>
          <p>Configure scenarios, run RL episodes, compare policies and review live plots. Best for analysis and demos.</p>
          <button className="btn btn-primary" onClick={onSimulator}>Launch simulator</button>
        </div>
        <div className="get-card">
          <div className="get-icon" aria-hidden>◉</div>
          <h3>Talk in Voice Mode</h3>
          <p>Describe a crisis out loud. Vishwamitra reflects, runs the simulation in the background, and replies with a verdict — in voice.</p>
          <button className="btn btn-secondary" onClick={onVoice}>Open voice mode</button>
        </div>
      </div>
    </section>
  )
}
