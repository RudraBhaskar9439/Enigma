export default function HowItWorks() {
  const steps = [
    {
      n: '01',
      title: 'Load a policy',
      body: 'Pick a meta-RL trained policy, a PPO baseline, or a random control. Loaded policies live in memory and drive every subsequent simulation.',
    },
    {
      n: '02',
      title: 'Design the crisis',
      body: 'Describe a school crisis in natural language. Vishwamitra routes it to the closest archetype — funding cut, teacher shortage, pandemic recovery, conflict — and applies your sliders.',
    },
    {
      n: '03',
      title: 'Choose your stakeholders',
      body: 'Select which voices should weigh in: Student, Teacher, Administrator, Policymaker. Each persona reasons from its own constraints.',
    },
    {
      n: '04',
      title: 'Run the episode',
      body: 'The RL policy interacts with the env step by step. Trajectories, rewards and intervention heatmaps stream into the right panel.',
    },
    {
      n: '05',
      title: 'Hear the verdict',
      body: 'A final verdict is composed from the stakeholder perspectives — short, opinionated, grounded in the simulated outcome.',
    },
    {
      n: '06',
      title: 'Teach Vishwamitra',
      body: 'Rate the verdict and leave a critique. The lesson is injected into the next run — you watch the answer improve in place.',
    },
  ]
  return (
    <section id="how" className="section alt">
      <div className="section-eyebrow">How it works</div>
      <h2 className="section-title">Six steps, one continuous learning loop</h2>
      <p className="section-lede">Vishwamitra closes the gap between policy simulation and human judgement. Here's the full pipeline.</p>
      <div className="steps">
        {steps.map(s => (
          <div key={s.n} className="step">
            <div className="step-n">{s.n}</div>
            <div>
              <h3>{s.title}</h3>
              <p>{s.body}</p>
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}
