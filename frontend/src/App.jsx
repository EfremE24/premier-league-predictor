import { useEffect, useMemo, useState } from 'react'
import './App.css'
import { FEATURE_GROUPS, FEATURE_LABELS, PRESETS, confidenceTier } from './content'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const OUTCOME_ORDER = ['Home win', 'Draw', 'Away win']
const MODEL_TYPE_LABELS = { RandomForestClassifier: 'Random Forest', LogisticRegression: 'Logistic Regression' }

function defaultDate() {
  const d = new Date()
  d.setDate(d.getDate() + 7)
  return d.toISOString().slice(0, 10)
}

function withValue(list, value) {
  return list.includes(value) ? list : [value, ...list]
}

function formatFixtureDate(isoDate) {
  const d = new Date(`${isoDate}T00:00:00`)
  return d.toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric' })
}

function App() {
  const [teams, setTeams] = useState([])
  const [teamsError, setTeamsError] = useState(null)
  const [modelInfo, setModelInfo] = useState(null)
  const [fixtures, setFixtures] = useState([])
  const [fixturesError, setFixturesError] = useState(null)
  const [selectedFixtureId, setSelectedFixtureId] = useState(null)

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

    fetch(`${API_URL}/model-info`)
      .then((res) => (res.ok ? res.json() : null))
      .then(setModelInfo)
      .catch(() => setModelInfo(null))

    fetch(`${API_URL}/fixtures`)
      .then((res) => {
        if (!res.ok) throw new Error(`GET /fixtures failed: ${res.status}`)
        return res.json()
      })
      .then((data) => {
        setFixtures(data.fixtures)
        if (data.error) setFixturesError(data.error)
      })
      .catch((err) => setFixturesError(err.message))
  }, [])

  const homeOptions = useMemo(() => withValue(teams, homeTeam), [teams, homeTeam])
  const awayOptions = useMemo(() => withValue(teams, awayTeam), [teams, awayTeam])

  // General odds-comparison search rather than one specific bookmaker --
  // adapts to whichever fixture is selected and doesn't imply an endorsement.
  const oddsSearchUrl = useMemo(() => {
    if (!homeTeam || !awayTeam) return null
    const query = `${homeTeam} vs ${awayTeam} odds`
    return `https://www.google.com/search?q=${encodeURIComponent(query)}`
  }, [homeTeam, awayTeam])

  function applyFixture(fixture) {
    setSelectedFixtureId(fixture.id)
    setHomeTeam(fixture.home_team)
    setAwayTeam(fixture.away_team)
    setDate(fixture.date)
    setResult(null)
    setError(null)
  }

  function applyPreset(preset) {
    setSelectedFixtureId(null)
    setHomeTeam(preset.home)
    setAwayTeam(preset.away)
    setAvgH(preset.avgH)
    setAvgD(preset.avgD)
    setAvgA(preset.avgA)
    setResult(null)
    setError(null)
  }

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

  const tier = result ? confidenceTier(result.probabilities) : null

  return (
    <div className="app">
      <header className="hero">
        <div className="hero-text">
          <p className="kicker">PL ML Project</p>
          <h1>Full Time</h1>
          <p className="subtitle">
            Premier League match outcome predictor. Home win / draw / away win probabilities
            from a Random Forest trained on 11 seasons of results and market odds.
          </p>
        </div>
        {modelInfo && (
          <div className="stat-pills">
            <div className="pill">
              <strong>{(modelInfo.train_row_count + modelInfo.test_row_count).toLocaleString()}</strong>
              <span>real matches</span>
            </div>
            <div className="pill">
              <strong>{modelInfo.test_log_loss.toFixed(3)}</strong>
              <span>log loss</span>
            </div>
            <div className="pill">
              <strong>{(modelInfo.test_accuracy * 100).toFixed(1)}%</strong>
              <span>accuracy</span>
            </div>
          </div>
        )}
      </header>

      {teamsError && (
        <p className="banner error">
          Could not load the team list from the API ({teamsError}). Is the backend running at{' '}
          <code>{API_URL}</code>?
        </p>
      )}

      <div className="layout">
        <section className="card">
          <h2>Matchup Inputs</h2>
          <p className="card-hint">Odds are decimal, average across bookmakers.</p>

          <div className="fixtures-block">
            <p className="presets-label">Upcoming fixtures (live from football-data.org)</p>
            {fixturesError && !fixtures.length && (
              <p className="fixtures-empty">Live fixtures unavailable right now -- pick a matchup manually below.</p>
            )}
            {!fixturesError && !fixtures.length && (
              <p className="fixtures-empty">Loading fixtures...</p>
            )}
            {fixtures.length > 0 && (
              <div className="fixtures-list">
                {fixtures.map((f) => (
                  <button
                    type="button"
                    key={f.id}
                    className={`fixture-row ${selectedFixtureId === f.id ? 'selected' : ''}`}
                    onClick={() => applyFixture(f)}
                  >
                    <span className="fixture-date">{formatFixtureDate(f.date)}</span>
                    <span className="fixture-matchup">{f.home_team} vs {f.away_team}</span>
                    {(!f.home_team_known || !f.away_team_known) && (
                      <span className="fixture-tag">cold start</span>
                    )}
                  </button>
                ))}
              </div>
            )}
          </div>

          <div className="presets">
            <p className="presets-label">Try an example matchup</p>
            <div className="presets-row">
              {PRESETS.map((p) => (
                <button type="button" key={p.label} className="chip" onClick={() => applyPreset(p)}>
                  {p.label}
                </button>
              ))}
            </div>
          </div>

          <form onSubmit={handleSubmit} className="predict-form">
            <div className="field-row">
              <label>
                Home team
                <select value={homeTeam} onChange={(e) => { setHomeTeam(e.target.value); setSelectedFixtureId(null) }} required>
                  {homeOptions.map((t) => (
                    <option key={t} value={t}>{t}</option>
                  ))}
                </select>
              </label>
              <label>
                Away team
                <select value={awayTeam} onChange={(e) => { setAwayTeam(e.target.value); setSelectedFixtureId(null) }} required>
                  {awayOptions.map((t) => (
                    <option key={t} value={t}>{t}</option>
                  ))}
                </select>
              </label>
            </div>

            <label>
              Match date
              <input type="date" value={date}
                     onChange={(e) => { setDate(e.target.value); setSelectedFixtureId(null) }} required />
            </label>

            <fieldset className="odds-fieldset">
              <legend>Market odds (decimal) &mdash; defaults provided, edit if you have real ones</legend>
              {oddsSearchUrl && (
                <a href={oddsSearchUrl} target="_blank" rel="noreferrer" className="odds-lookup-link">
                  Need current odds? Find {homeTeam} vs {awayTeam} odds &rarr;
                </a>
              )}
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

            <button type="submit" className="submit" disabled={loading || teams.length === 0}>
              {loading ? 'Predicting...' : 'Predict Matchup'}
            </button>
          </form>

          {error && <p className="banner error">{error}</p>}
        </section>

        <section className="card">
          <h2>Prediction</h2>
          {!result && <p className="card-hint">Fill in a matchup and predict to see results here.</p>}

          {result && tier && (
            <>
              <div className="prediction-head">
                <p className="matchup-line">{result.home_team} vs {result.away_team}</p>
                <span className={`badge tone-${tier.tone}`}>{tier.label}</span>
              </div>
              <p className="result-meta">{result.date} &middot; {result.season}</p>

              <div className="model-pick">
                <span className="model-pick-label">Model pick</span>
                <span className="model-pick-value">{result.predicted_outcome}</span>
              </div>

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
            </>
          )}
        </section>
      </div>

      {result && (
        <section className="card features-used">
          <h2>Model Features Used</h2>
          <p className="card-hint">
            The 15 values actually sent to the model for this prediction, computed from team
            history as of right before this fixture.
          </p>
          {FEATURE_GROUPS.map((group) => (
            <div key={group.group} className="feature-group">
              <p className="feature-group-label">{group.group}</p>
              <div className="feature-grid">
                {group.keys.map((key) => (
                  <div key={key} className="feature-cell">
                    <span className="feature-value">{result.features[key]}</span>
                    <span className="feature-label">{FEATURE_LABELS[key]}</span>
                    <span className="feature-key">{key}</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </section>
      )}

      <section className="card how-to-read">
        <h2>How To Read It</h2>
        <p className="card-hint">The prediction is a probability estimate, not a guaranteed outcome.</p>
        <div className="read-cards">
          <div className="read-card">
            <p className="read-title">Draws are genuinely hard</p>
            <p>
              About 1 in 4 Premier League matches end in a draw. The model puts real probability
              mass on draws rather than avoiding the prediction, but it's the hardest class to
              call -- see the calibration write-up in the repo for specifics.
            </p>
          </div>
          <div className="read-card">
            <p className="read-title">The market matters most</p>
            <p>
              The two market-odds features are the largest signal this model uses (see Feature
              Importance below) -- more than any single team-strength feature. That matches
              published findings that betting markets are hard to beat.
            </p>
          </div>
          <div className="read-card">
            <p className="read-title">Educational use only</p>
            <p>
              This is a portfolio ML project trained on historical results. It is not betting
              advice.
            </p>
          </div>
        </div>
      </section>

      {modelInfo && (
        <section className="card">
          <h2>Feature Importance</h2>
          <p className="card-hint">
            How much the trained Random Forest actually relies on each input, across all
            predictions -- not specific to the matchup above.
          </p>
          <ul className="importance-list">
            {modelInfo.feature_importances.map((row) => {
              const max = modelInfo.feature_importances[0].importance
              return (
                <li key={row.feature}>
                  <span className="label">{FEATURE_LABELS[row.feature] ?? row.feature}</span>
                  <div className="bar-track">
                    <div className="bar-fill" style={{ width: `${(row.importance / max) * 100}%` }} />
                  </div>
                  <span className="pct">{row.importance.toFixed(3)}</span>
                </li>
              )
            })}
          </ul>
        </section>
      )}

      {modelInfo && (
        <section className="card model-metrics">
          <h2>Model Metrics</h2>
          <p className="card-hint">Time-based train/test split on real Premier League seasons -- never random.</p>
          <div className="metrics-grid">
            <div className="metric">
              <span className="metric-value">{MODEL_TYPE_LABELS[modelInfo.model_type] ?? modelInfo.model_type}</span>
              <span className="metric-label">Model</span>
            </div>
            <div className="metric">
              <span className="metric-value">{modelInfo.train_row_count.toLocaleString()}</span>
              <span className="metric-label">Train rows ({modelInfo.train_seasons[0]}-{modelInfo.train_seasons.at(-1)})</span>
            </div>
            <div className="metric">
              <span className="metric-value">{modelInfo.test_row_count.toLocaleString()}</span>
              <span className="metric-label">Test rows ({modelInfo.test_seasons.join(', ')})</span>
            </div>
            <div className="metric">
              <span className="metric-value">{(modelInfo.test_accuracy * 100).toFixed(1)}%</span>
              <span className="metric-label">Test accuracy</span>
            </div>
            <div className="metric">
              <span className="metric-value">{modelInfo.test_log_loss.toFixed(4)}</span>
              <span className="metric-label">Test log loss (selection metric)</span>
            </div>
          </div>
        </section>
      )}

      <footer>
        <p>Data: football-data.co.uk. Portfolio project -- not betting advice.</p>
      </footer>
    </div>
  )
}

export default App
