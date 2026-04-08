const BASE = import.meta.env.VITE_BACKEND_URL || 'http://localhost:7860'

async function post(path, body) {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  })
  if (!res.ok) throw new Error(`${path} → ${res.status}`)
  return res.json()
}

export const api = {
  loadPolicy: (model_type) => post('/api/load_policy', { model_type }),
  createScenario: (payload) => post('/api/create_scenario', payload),
  runSimulation: (payload) => post('/api/run_simulation', payload),
  feedback: (payload) => post('/api/feedback', payload),
  compare: (n_steps) => post('/api/compare', { n_steps }),
}
