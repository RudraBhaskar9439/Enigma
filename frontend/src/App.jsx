import { useState } from 'react'
import Navbar from './components/Navbar.jsx'
import Hero from './components/Hero.jsx'
import Features from './components/Features.jsx'
import HowItWorks from './components/HowItWorks.jsx'
import GetStarted from './components/GetStarted.jsx'
import Footer from './components/Footer.jsx'
import Simulator from './components/Simulator.jsx'
import VoiceSession from './components/VoiceSession.jsx'

export default function App() {
  const [view, setView] = useState('home') // home | simulator | voice

  const goHome = () => setView('home')
  const goSim = () => setView('simulator')
  const goVoice = () => setView('voice')

  return (
    <div className="app">
      <Navbar onHome={goHome} onLaunch={goSim} onNavigate={goHome} />

      {view === 'home' && (
        <>
          <Hero onStart={goSim} />
          <Features />
          <HowItWorks />
          <GetStarted onSimulator={goSim} onVoice={goVoice} />
        </>
      )}

      {view !== 'home' && (
        <div className="modetabs">
          <button onClick={goHome}>← Home</button>
          <button className={view === 'simulator' ? 'on' : ''} onClick={goSim}>Simulator</button>
          <button className={view === 'voice' ? 'on' : ''} onClick={goVoice}>Voice</button>
        </div>
      )}

      {view === 'simulator' && <Simulator />}
      {view === 'voice' && <VoiceSession onBack={goHome} />}

      <Footer />
    </div>
  )
}
