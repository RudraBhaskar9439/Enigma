export default function Hero({ onStart }) {
  return (
    <section className="hero">
      <span className="hero-eyebrow">● Meta · PyTorch Hackathon — Agentic Education Simulator</span>
      <h1>
        Stress-test education systems with <span className="grad">Vishwamitra</span>
      </h1>
      <p className="lede">
        An agentic, meta-RL crisis simulator. Configure a scenario, hear from
        student / teacher / admin / policymaker LLM personas, run a policy, and
        teach Vishwamitra with your feedback — in voice or text.
      </p>
      <div className="hero-cta">
        <button className="btn btn-primary" onClick={onStart}>Open simulator</button>
        <a className="btn btn-secondary" href="#how">How it works</a>
      </div>
    </section>
  )
}
