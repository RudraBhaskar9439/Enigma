import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import PlotMod from 'react-plotly.js'
import { api } from '../lib/api.js'

const Plot = PlotMod.default || PlotMod

const ALL_AGENTS = ['Student', 'Teacher', 'Administrator', 'Policymaker']

export default function Simulator() {
  // 01 Load Policy
  const [modelType, setModelType] = useState('meta_rl')
  const [loadStatus, setLoadStatus] = useState('')

  // 02 Scenario
  const [crisisText, setCrisisText] = useState('')
  const [difficulty, setDifficulty] = useState('medium')
  const [initialBudget, setInitialBudget] = useState(70)
  const [teacherRetention, setTeacherRetention] = useState(75)
  const [enrollmentRate, setEnrollmentRate] = useState(85)
  const [scenarioStatus, setScenarioStatus] = useState('')

  // 03 Stakeholders
  const [agents, setAgents] = useState([...ALL_AGENTS])

  // 04 Run
  const [nSteps, setNSteps] = useState(100)
  const [useInterventions, setUseInterventions] = useState(true)

  // Outputs
  const [perspectives, setPerspectives] = useState('_Run a simulation to hear from your stakeholders._')
  const [verdict, setVerdict] = useState('')
  const [statusMd, setStatusMd] = useState('')
  const [trajPlot, setTrajPlot] = useState(null)
  const [metricsPlot, setMetricsPlot] = useState(null)
  const [intervPlot, setIntervPlot] = useState(null)
  const [activePlotTab, setActivePlotTab] = useState('traj')

  // 05 Feedback
  const [fbRating, setFbRating] = useState(3)
  const [fbComment, setFbComment] = useState('')

  // Compare
  const [compareSteps, setCompareSteps] = useState(50)
  const [compareReport, setCompareReport] = useState('')
  const [comparePlot, setComparePlot] = useState(null)

  // Loading flags
  const [loading, setLoading] = useState({})
  const setBusy = (k, v) => setLoading(s => ({ ...s, [k]: v }))

  function applyResult(r) {
    setPerspectives(r.perspectives_md || '')
    setVerdict(r.verdict_md || '')
    setStatusMd(r.status_md || '')
    setTrajPlot(r.trajectory_plot || null)
    setMetricsPlot(r.metrics_plot || null)
    setIntervPlot(r.intervention_plot || null)
  }

  async function handleLoad() {
    setBusy('load', true)
    try { const r = await api.loadPolicy(modelType); setLoadStatus(r.status) }
    catch (e) { setLoadStatus(`❌ ${e.message}`) }
    finally { setBusy('load', false) }
  }
  async function handleCreate() {
    setBusy('create', true)
    try {
      const r = await api.createScenario({
        crisis_text: crisisText, difficulty,
        initial_budget: initialBudget, teacher_retention: teacherRetention, enrollment_rate: enrollmentRate,
      })
      setScenarioStatus(r.status)
    } catch (e) { setScenarioStatus(`❌ ${e.message}`) }
    finally { setBusy('create', false) }
  }
  async function handleRun() {
    setBusy('run', true)
    try {
      const r = await api.runSimulation({ selected_agents: agents, n_steps: nSteps, use_interventions: useInterventions })
      applyResult(r)
    } catch (e) { setStatusMd(`❌ ${e.message}`) }
    finally { setBusy('run', false) }
  }
  async function handleFeedback() {
    setBusy('fb', true)
    try { const r = await api.feedback({ rating: fbRating, comment: fbComment }); applyResult(r); setFbComment('') }
    catch (e) { setStatusMd(`❌ ${e.message}`) }
    finally { setBusy('fb', false) }
  }
  async function handleCompare() {
    setBusy('cmp', true)
    try { const r = await api.compare(compareSteps); setCompareReport(r.report || ''); setComparePlot(r.plot || null) }
    catch (e) { setCompareReport(`❌ ${e.message}`) }
    finally { setBusy('cmp', false) }
  }

  function toggleAgent(a) {
    setAgents(prev => prev.includes(a) ? prev.filter(x => x !== a) : [...prev, a])
  }

  return (
    <section className="sim">
      <div className="sim-grid">
        {/* LEFT CONTROL COLUMN */}
        <div className="sim-controls">
          <Card title="01 · Load Policy">
            <label className="lbl">Policy</label>
            <select className="input" value={modelType} onChange={e => setModelType(e.target.value)}>
              <option value="meta_rl">meta_rl</option>
              <option value="ppo_standard">ppo_standard</option>
              <option value="random">random</option>
            </select>
            <button className="btn btn-primary full" disabled={loading.load} onClick={handleLoad}>
              {loading.load ? 'Loading…' : 'Load Policy'}
            </button>
            {loadStatus && <div className="status">{loadStatus}</div>}
          </Card>

          <Card title="02 · Configure Scenario">
            <label className="lbl">Crisis archetype</label>
            <textarea
              className="input" rows="3"
              placeholder='Describe the crisis — e.g. "sudden 40% budget cut after audit"'
              value={crisisText} onChange={e => setCrisisText(e.target.value)}
            />
            <label className="lbl">Difficulty</label>
            <select className="input" value={difficulty} onChange={e => setDifficulty(e.target.value)}>
              <option>easy</option><option>medium</option><option>hard</option>
            </select>
            <Slider label="Initial Budget (%)" min={30} max={100} value={initialBudget} onChange={setInitialBudget} />
            <Slider label="Teacher Retention (%)" min={30} max={100} value={teacherRetention} onChange={setTeacherRetention} />
            <Slider label="Initial Enrollment (%)" min={50} max={100} value={enrollmentRate} onChange={setEnrollmentRate} />
            <button className="btn btn-secondary full" disabled={loading.create} onClick={handleCreate}>
              {loading.create ? 'Creating…' : 'Create Scenario'}
            </button>
            {scenarioStatus && <pre className="status pre">{scenarioStatus}</pre>}
          </Card>

          <Card title="03 · Choose Stakeholders">
            <div className="checks">
              {ALL_AGENTS.map(a => (
                <label key={a} className={`check ${agents.includes(a) ? 'on' : ''}`}>
                  <input type="checkbox" checked={agents.includes(a)} onChange={() => toggleAgent(a)} />
                  <span>{a}</span>
                </label>
              ))}
            </div>
          </Card>

          <Card title="04 · Run Episode">
            <Slider label="Episode Length" min={50} max={200} step={10} value={nSteps} onChange={setNSteps} />
            <label className="check" style={{ marginTop: 8 }}>
              <input type="checkbox" checked={useInterventions} onChange={e => setUseInterventions(e.target.checked)} />
              <span>Enable mechanism-design interventions</span>
            </label>
            <button className="btn btn-primary full" disabled={loading.run} onClick={handleRun}>
              {loading.run ? 'Running…' : 'Run Simulation'}
            </button>
          </Card>

          <Card title="05 · Teach Vishwamitra">
            <p className="muted">Rate the verdict, leave a note, and re-run with the lesson injected.</p>
            <Slider label={`Verdict quality: ${fbRating}/5`} min={1} max={5} step={1} value={fbRating} onChange={setFbRating} />
            <textarea
              className="input" rows="3"
              placeholder="What would you change?"
              value={fbComment} onChange={e => setFbComment(e.target.value)}
            />
            <button className="btn btn-secondary full" disabled={loading.fb} onClick={handleFeedback}>
              {loading.fb ? 'Submitting…' : 'Submit Feedback & Retry'}
            </button>
          </Card>

          <Card title="Compare Policies">
            <Slider label="Episode Length" min={20} max={100} step={10} value={compareSteps} onChange={setCompareSteps} />
            <button className="btn btn-primary full" disabled={loading.cmp} onClick={handleCompare}>
              {loading.cmp ? 'Comparing…' : 'Run Comparison'}
            </button>
            {compareReport && <pre className="status pre">{compareReport}</pre>}
            {comparePlot && (
              <div className="plot-wrap" style={{ marginTop: 12 }}>
                <Plot data={comparePlot.data} layout={{ ...comparePlot.layout, autosize: true }} useResizeHandler style={{ width: '100%', height: '320px' }} config={{ displayModeBar: false }} />
              </div>
            )}
          </Card>
        </div>

        {/* RIGHT OUTPUT COLUMN */}
        <div className="sim-output">
          <div className="output-card">
            <ReactMarkdown>{perspectives}</ReactMarkdown>
          </div>
          {verdict && (
            <div className="output-card verdict">
              <ReactMarkdown>{verdict}</ReactMarkdown>
            </div>
          )}
          {statusMd && (
            <div className="output-card">
              <ReactMarkdown>{statusMd}</ReactMarkdown>
            </div>
          )}

          <div className="plot-card">
            <div className="plot-tabs">
              <button className={activePlotTab === 'traj' ? 'on' : ''} onClick={() => setActivePlotTab('traj')}>Crisis Trajectories</button>
              <button className={activePlotTab === 'metrics' ? 'on' : ''} onClick={() => setActivePlotTab('metrics')}>Reward & Health</button>
              <button className={activePlotTab === 'interv' ? 'on' : ''} onClick={() => setActivePlotTab('interv')}>Intervention Heatmap</button>
            </div>
            <div className="plot-wrap">
              {activePlotTab === 'traj' && trajPlot && <Plot data={trajPlot.data} layout={{ ...trajPlot.layout, autosize: true }} useResizeHandler style={{ width: '100%', height: '520px' }} config={{ displayModeBar: false }} />}
              {activePlotTab === 'metrics' && metricsPlot && <Plot data={metricsPlot.data} layout={{ ...metricsPlot.layout, autosize: true }} useResizeHandler style={{ width: '100%', height: '340px' }} config={{ displayModeBar: false }} />}
              {activePlotTab === 'interv' && intervPlot && <Plot data={intervPlot.data} layout={{ ...intervPlot.layout, autosize: true }} useResizeHandler style={{ width: '100%', height: '400px' }} config={{ displayModeBar: false }} />}
              {((activePlotTab === 'traj' && !trajPlot) || (activePlotTab === 'metrics' && !metricsPlot) || (activePlotTab === 'interv' && !intervPlot)) && (
                <div className="plot-empty">Run a simulation to see this chart.</div>
              )}
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

function Card({ title, children }) {
  return (
    <div className="ctrl-card">
      <div className="ctrl-title">{title}</div>
      {children}
    </div>
  )
}

function Slider({ label, min, max, step = 1, value, onChange }) {
  return (
    <div className="slider">
      <div className="slider-row">
        <span className="lbl">{label}</span>
        <span className="slider-val">{value}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value} onChange={e => onChange(Number(e.target.value))} />
    </div>
  )
}
