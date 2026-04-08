export default function Features() {
  const items = [
    {
      icon: '✦',
      title: 'Multi-Stakeholder LLM Agents',
      body: 'Hear from Student, Teacher, Administrator and Policymaker personas. Every voice is consulted before a verdict is delivered.',
    },
    {
      icon: '◐',
      title: 'Empathic Voice Mode',
      body: 'Speak naturally to Vishwamitra. The agent listens, reflects, and responds with an emotionally intelligent voice.',
    },
    {
      icon: '↻',
      title: 'Feedback-Driven Learning',
      body: 'Disagree with a verdict? Submit feedback and Vishwamitra re-runs the simulation with your lesson injected — improving in place.',
    },
    {
      icon: '◭',
      title: 'Live Trajectory Plots',
      body: 'Watch enrollment, attendance, retention, dropout and intervention intensity evolve over the episode in real time.',
    },
    {
      icon: '⚖',
      title: 'Compare Policies',
      body: 'Benchmark the meta-trained policy against random and PPO baselines on the same scenario, side by side.',
    },
  ]
  return (
    <section id="features" className="section">
      <div className="section-eyebrow">Features</div>
      <h2 className="section-title">Everything you need to stress-test an education system</h2>
      <p className="section-lede">Six tools, one cohesive workflow — from crisis design to RL execution to human-in-the-loop refinement.</p>
      <div className="feat-grid">
        {items.map(it => (
          <div key={it.title} className="feat-card">
            <div className="feat-icon">{it.icon}</div>
            <h3>{it.title}</h3>
            <p>{it.body}</p>
          </div>
        ))}
      </div>
    </section>
  )
}
