import { useEffect, useMemo, useState } from 'react'
import './App.css'
import { FEATURE_GROUPS, FEATURE_LABELS, PRESETS, TEAM_COLORS, confidenceTier } from './content'

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

// "Home win" / "Away win" on their own don't say which actual team that
// is -- this turns them into "Liverpool (Home)" / "Liverpool (Away)" so
// the probability rows and Model Pick read as a real team, not just a
// role. Draw has no team to attach, so it stays "Draw".
function outcomeDisplayLabel(outcomeLabel, homeTeam, awayTeam) {
  if (outcomeLabel === 'Home win') return `${homeTeam} (Home)`
  if (outcomeLabel === 'Away win') return `${awayTeam} (Away)`
  return 'Draw'
}

function monthKey(isoDate) {
  return isoDate.slice(0, 7) // "2026-09-04" -> "2026-09"
}

function formatMonthLabel(key) {
  const d = new Date(`${key}-01T00:00:00`)
  return d.toLocaleDateString(undefined, { month: 'long', year: 'numeric' })
}

// Split circle, primary color on the left half and secondary on the
// right -- a team not in TEAM_COLORS (a one-off preset opponent, say)
// gets a plain neutral gray rather than breaking.
function TeamSwatch({ team }) {
  const [primary, secondary] = TEAM_COLORS[team] ?? ['#c7c2b8', '#c7c2b8']
  return (
    <span
      className="team-swatch"
      style={{ background: `linear-gradient(90deg, ${primary} 50%, ${secondary} 50%)` }}
      title={team}
      aria-hidden="true"
    />
  )
}

const THEME_KEY = 'pl-predictor-theme'

function getSystemTheme() {
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

function getStoredTheme() {
  try {
    return localStorage.getItem(THEME_KEY)
  } catch {
    return null // localStorage can throw in private-browsing/locked-down contexts
  }
}

function App() {
  const [theme, setTheme] = useState(() => getStoredTheme() ?? getSystemTheme())

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    try {
      localStorage.setItem(THEME_KEY, theme)
    } catch {
      // per-visitor convenience only -- fine if it can't persist
    }
  }, [theme])

  const [teams, setTeams] = useState([])
  const [teamsError, setTeamsError] = useState(null)
  const [modelInfo, setModelInfo] = useState(null)
  const [fixtures, setFixtures] = useState([])
  const [fixturesError, setFixturesError] = useState(null)
  const [selectedFixtureId, setSelectedFixtureId] = useState(null)
  const [selectedMonth, setSelectedMonth] = useState(null)

  const [mode, setMode] = useState('combined')
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
        // Default to the first not-yet-played match's month, not literally
        // fixtures[0] -- the list now includes the whole season, so
        // fixtures[0] is whatever's earliest overall (often already
        // finished), and landing there on load would bury the actually
        // relevant "what can I predict" view under already-played months.
        const firstUpcoming = data.fixtures.find((f) => f.status !== 'FINISHED') ?? data.fixtures[0]
        if (firstUpcoming) setSelectedMonth(monthKey(firstUpcoming.date))
        if (data.error) setFixturesError(data.error)
      })
      .catch((err) => setFixturesError(err.message))
  }, [])

  const homeOptions = useMemo(() => withValue(teams, homeTeam), [teams, homeTeam])
  const awayOptions = useMemo(() => withValue(teams, awayTeam), [teams, awayTeam])

  // All months that actually have a fixture in them, earliest first --
  // not a fixed Aug-May list, so this still makes sense mid-season when
  // only some months have anything left to show.
  const availableMonths = useMemo(
    () => [...new Set(fixtures.map((f) => monthKey(f.date)))].sort(),
    [fixtures],
  )
  const visibleFixtures = useMemo(
    () => (selectedMonth ? fixtures.filter((f) => monthKey(f.date) === selectedMonth) : fixtures),
    [fixtures, selectedMonth],
  )

  // General odds-comparison search rather than one specific bookmaker --
  // adapts to whichever fixture is selected and doesn't imply an endorsement.
  const oddsSearchUrl = useMemo(() => {
    if (!homeTeam || !awayTeam) return null
    const query = `${homeTeam} vs ${awayTeam} odds decimal`
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
          mode,
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
        <div className="hero-top">
          <p className="kicker">PL ML Project</p>
          <button
            type="button"
            className="theme-toggle"
            onClick={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))}
          >
            {theme === 'dark' ? 'Light mode' : 'Dark mode'}
          </button>
        </div>
        <h1>Full Time</h1>
        <p className="subtitle">
          Predicts home win, draw, or away win for a Premier League fixture. Trained on 11
          seasons of results and closing market odds -- the numbers are at the bottom of the page.
        </p>
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

          {modelInfo && (
            <div className="mode-select">
              <p className="block-label">Prediction mode</p>
              <div className="mode-options">
                {Object.entries(modelInfo.modes).map(([key, info]) => (
                  <label key={key} className={`mode-option ${mode === key ? 'selected' : ''}`}>
                    <input
                      type="radio"
                      name="mode"
                      value={key}
                      checked={mode === key}
                      onChange={() => { setMode(key); setResult(null); setError(null) }}
                    />
                    <span className="mode-option-label">{info.label}</span>
                    <span className="mode-option-desc">{info.description}</span>
                  </label>
                ))}
              </div>
            </div>
          )}

          <div className="fixtures-block">
            <div className="fixtures-heading">
              <p className="block-label">This season, from football-data.org</p>
              {availableMonths.length > 1 && (
                <select
                  className="month-filter"
                  value={selectedMonth ?? ''}
                  onChange={(e) => setSelectedMonth(e.target.value)}
                >
                  {availableMonths.map((m) => (
                    <option key={m} value={m}>{formatMonthLabel(m)}</option>
                  ))}
                </select>
              )}
            </div>
            {fixturesError && !fixtures.length && (
              <p className="fixtures-empty">Fixtures aren't loading right now -- pick a matchup by hand below.</p>
            )}
            {!fixturesError && !fixtures.length && (
              <p className="fixtures-empty">Loading...</p>
            )}
            {visibleFixtures.length > 0 && (
              <div className="fixtures-list">
                {visibleFixtures.map((f) => {
                  const isFinished = f.status === 'FINISHED'
                  const matchupContent = (
                    <>
                      <span className="fixture-date">{formatFixtureDate(f.date)}</span>
                      <span className="fixture-matchup">
                        <TeamSwatch team={f.home_team} /> {f.home_team} vs{' '}
                        <TeamSwatch team={f.away_team} /> {f.away_team}
                      </span>
                      {isFinished ? (
                        <span className="fixture-score">{f.home_score}&ndash;{f.away_score} FT</span>
                      ) : (
                        (!f.home_team_known || !f.away_team_known) && (
                          <span className="fixture-tag">cold start</span>
                        )
                      )}
                    </>
                  )
                  return isFinished ? (
                    <div
                      key={f.id}
                      className="fixture-row finished"
                      title="Already played -- team-strength data is frozen as of training time, so this isn't offered as something to predict."
                    >
                      {matchupContent}
                    </div>
                  ) : (
                    <button
                      type="button"
                      key={f.id}
                      className={`fixture-row ${selectedFixtureId === f.id ? 'selected' : ''}`}
                      onClick={() => applyFixture(f)}
                    >
                      {matchupContent}
                    </button>
                  )
                })}
              </div>
            )}
          </div>

          <div className="presets">
            <span className="block-label">or try:</span>{' '}
            {PRESETS.map((p, i) => (
              <span key={p.label}>
                <button type="button" className="text-link" onClick={() => applyPreset(p)}>{p.label}</button>
                {i < PRESETS.length - 1 && <span className="dot-sep">&middot;</span>}
              </span>
            ))}
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
              <legend>Odds (decimal)</legend>
              {oddsSearchUrl && (
                <a href={oddsSearchUrl} target="_blank" rel="noreferrer" className="odds-lookup-link">
                  Look up {homeTeam} vs {awayTeam} (decimal) &rarr;
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
          {!result && <p className="card-hint">Nothing yet -- fill in a matchup.</p>}

          {result && tier && (
            <>
              <div className="prediction-head">
                <p className="matchup-line">
                  <TeamSwatch team={result.home_team} /> {result.home_team} vs{' '}
                  <TeamSwatch team={result.away_team} /> {result.away_team}
                </p>
                <span className={`badge tone-${tier.tone}`}>{tier.label}</span>
              </div>
              <p className="result-meta">{result.date} &middot; {result.season}</p>

              <div className="model-pick">
                <span className="model-pick-label">Model pick</span>
                <span className="model-pick-value">
                  {outcomeDisplayLabel(result.predicted_outcome, result.home_team, result.away_team)}
                </span>
              </div>

              <ul className="probabilities">
                {OUTCOME_ORDER.map((label) => {
                  const prob = result.probabilities[label]
                  const isWinner = label === result.predicted_outcome
                  const rowTeam = label === 'Home win' ? result.home_team
                    : label === 'Away win' ? result.away_team : null
                  return (
                    <li key={label} className={isWinner ? 'winner' : ''}>
                      <span className="label">
                        {rowTeam && <TeamSwatch team={rowTeam} />}{' '}
                        {outcomeDisplayLabel(label, result.home_team, result.away_team)}
                      </span>
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

      {result && modelInfo && (() => {
        const usedCols = modelInfo.modes[result.mode]?.feature_columns ?? []
        return (
          <section className="card features-used">
            <h2>What the model actually saw</h2>
            <p className="card-hint">
              All 15 computed values, as of right before kickoff. {modelInfo.modes[result.mode]?.label} mode
              only fed the highlighted ones into the model -- the rest are shown for context, grayed out.
            </p>
            {FEATURE_GROUPS.map((group) => (
              <div key={group.group} className="feature-group">
                <p className="feature-group-label">{group.group}</p>
                <div className="feature-grid">
                  {group.keys.map((key) => (
                    <div key={key} className={`feature-cell ${usedCols.includes(key) ? '' : 'unused'}`}>
                      <span className="feature-value">{result.features[key]}</span>
                      <span className="feature-label">{FEATURE_LABELS[key]}</span>
                      <span className="feature-key">{key}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </section>
        )
      })()}

      <section className="editorial">
        <h2>Before you trust the number</h2>
        <ol className="read-list">
          <li>
            <strong>Draws are genuinely hard.</strong> About 1 in 4 Premier League matches end
            level. The model puts real weight on a draw rather than dodging the call, but it's
            the outcome it gets wrong most -- see the calibration write-up in the repo.
          </li>
          <li>
            <strong>The market is doing most of the work.</strong> The two market-odds inputs
            outweigh every team-strength feature combined (below). That tracks with the broader
            finding that betting markets are hard to beat -- this model mostly agrees with them,
            it doesn't outsmart them.
          </li>
          <li>
            <strong>This is a portfolio project, not a tipster.</strong> Historical results,
            trained for practice. Don't bet on it.
          </li>
        </ol>
      </section>

      {modelInfo && (
        <section className="card">
          <h2>What the model leans on</h2>
          <p className="card-hint">
            {modelInfo.modes[mode].label} mode, across every prediction it's made -- not just the one above.
            Switch modes to compare.
          </p>
          <ul className="importance-list">
            {modelInfo.modes[mode].feature_importances.map((row) => {
              const max = modelInfo.modes[mode].feature_importances[0].importance
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
          <h2>The fine print</h2>
          <p className="card-hint">
            {modelInfo.modes[mode].label} mode. Time-based split -- trained on earlier seasons, tested on
            later ones. Never random.
          </p>
          <div className="metrics-grid">
            <div className="metric">
              <span className="metric-value">
                {MODEL_TYPE_LABELS[modelInfo.modes[mode].model_type] ?? modelInfo.modes[mode].model_type}
              </span>
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
              <span className="metric-value">{(modelInfo.modes[mode].test_accuracy * 100).toFixed(1)}%</span>
              <span className="metric-label">Test accuracy</span>
            </div>
            <div className="metric">
              <span className="metric-value">{modelInfo.modes[mode].test_log_loss.toFixed(4)}</span>
              <span className="metric-label">Test log loss (selection metric)</span>
            </div>
            <div className="metric">
              <span className="metric-value">{modelInfo.modes[mode].test_roc_auc.toFixed(4)}</span>
              <span className="metric-label">Test ROC AUC</span>
            </div>
            <div className="metric">
              <span className="metric-value">{modelInfo.modes[mode].test_brier_score.toFixed(4)}</span>
              <span className="metric-label">Test Brier score</span>
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
