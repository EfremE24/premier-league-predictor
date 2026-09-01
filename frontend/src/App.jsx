import { useEffect, useState } from 'react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const OUTCOME_ORDER = ['Home win', 'Draw', 'Away win']

function defaultDate() {
  const d = new Date()
  d.setDate(d.getDate() + 7)
  return d.toISOString().slice(0, 10)
}

function App() {
  const [teams, setTeams] = useState([])
  const [teamsError, setTeamsError] = useState(null)
  const [homeTeam, setHomeTeam] = useState('')
  const [awayTeam, setAwayTeam] = useState('')
  const [date, setDate] = useState(defaultDate())
  const [avgH, setAvgH] = useState('2.50')
  const [avgD, setAvgD] = useState('3.40')
  const [avgA, setAvgA] = useState('2.90')
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    fetch(`${API_URL}/teams`)
      .then((res) => {
        if (!res.ok) throw new Error(`GET /teams failed: ${res.status}`)
        return res.json()
      })
      .then((data) => {
        setTeams(data.teams)
        if (data.teams.length >= 2) {
          setHomeTeam(data.teams[0])
          setAwayTeam(data.teams[1])
        }
      })
      .catch((err) => setTeamsError(err.message))
  }, [])

  async function handleSubmit(e) {
    e.preventDefault()
    setError(null)
    setResult(null)

    if (homeTeam === awayTeam) {
      setError('Home and away team must be different.')
      return
    }

    setLoading(true)
    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          home_team: homeTeam,
          away_team: awayTeam,
          date,
          avg_h: parseFloat(avgH),
          avg_d: parseFloat(avgD),
          avg_a: parseFloat(avgA),
        }),
      })
      const data = await res.json()
      if (!res.ok) {
        const detail = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail)
        throw new Error(detail || `Request failed: ${res.status}`)
      }
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header>
        <h1>Premier League Match Predictor</h1>
        <p className="subtitle">
          Home win / draw / away win probabilities from a Random Forest model trained on 11
          seasons of results and market odds.
        </p>
      </header>

      {teamsError && (
        <p className="error">
          Could not load the team list from the API ({teamsError}). Is the backend running at{' '}
          <code>{API_URL}</code>?
        </p>
      )}

      <form onSubmit={handleSubmit} className="predict-form">
        <div className="field-row">
          <label>
            Home team
            <select value={homeTeam} onChange={(e) => setHomeTeam(e.target.value)} required>
              {teams.map((t) => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
          </label>
          <label>
            Away team
            <select value={awayTeam} onChange={(e) => setAwayTeam(e.target.value)} required>
              {teams.map((t) => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
          </label>
        </div>

        <label>
          Match date
          <input type="date" value={date} onChange={(e) => setDate(e.target.value)} required />
        </label>

        <fieldset className="odds-fieldset">
          <legend>Market odds (decimal) &mdash; defaults provided, edit if you have real ones</legend>
          <div className="field-row">
            <label>
              Home win
              <input type="number" step="0.01" min="1.01" value={avgH}
                     onChange={(e) => setAvgH(e.target.value)} required />
            </label>
            <label>
              Draw
              <input type="number" step="0.01" min="1.01" value={avgD}
                     onChange={(e) => setAvgD(e.target.value)} required />
            </label>
            <label>
              Away win
              <input type="number" step="0.01" min="1.01" value={avgA}
                     onChange={(e) => setAvgA(e.target.value)} required />
            </label>
          </div>
        </fieldset>

        <button type="submit" disabled={loading || teams.length === 0}>
          {loading ? 'Predicting...' : 'Predict'}
        </button>
      </form>

      {error && <p className="error">{error}</p>}

      {result && (
        <section className="result">
          <h2>{result.home_team} vs {result.away_team}</h2>
          <p className="result-meta">{result.date} &middot; {result.season}</p>
          <ul className="probabilities">
            {OUTCOME_ORDER.map((label) => {
              const prob = result.probabilities[label]
              const isWinner = label === result.predicted_outcome
              return (
                <li key={label} className={isWinner ? 'winner' : ''}>
                  <span className="label">{label}</span>
                  <div className="bar-track">
                    <div className="bar-fill" style={{ width: `${prob * 100}%` }} />
                  </div>
                  <span className="pct">{(prob * 100).toFixed(1)}%</span>
                </li>
              )
            })}
          </ul>
          <p className="predicted">
            Most likely outcome: <strong>{result.predicted_outcome}</strong>
          </p>
        </section>
      )}
    </div>
  )
}

export default App
