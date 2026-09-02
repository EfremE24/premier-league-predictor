// Static presentational data: friendly labels for the raw feature keys the
// API returns, and a few quick-fill presets for the form. Kept out of
// App.jsx so the component stays about behavior, not this kind of list.

export const FEATURE_GROUPS = [
  {
    group: 'Elo rating',
    keys: ['elo_home', 'elo_away', 'elo_gap'],
  },
  {
    group: 'Recent form (PPG, last 5)',
    keys: ['form_home', 'form_away', 'form_gap'],
  },
  {
    group: 'Home / away split',
    keys: ['home_ppg', 'away_ppg', 'home_advantage_gap'],
  },
  {
    group: 'Rest',
    keys: ['rest_days_home', 'rest_days_away', 'rest_days_gap'],
  },
  {
    group: 'Market-implied probability',
    keys: ['imp_prob_h', 'imp_prob_d', 'imp_prob_a'],
  },
]

export const FEATURE_LABELS = {
  elo_home: 'Home team Elo',
  elo_away: 'Away team Elo',
  elo_gap: 'Elo gap',
  form_home: 'Home team form',
  form_away: 'Away team form',
  form_gap: 'Form gap',
  home_ppg: "Home team's home PPG",
  away_ppg: "Away team's away PPG",
  home_advantage_gap: 'Home-advantage gap',
  rest_days_home: 'Home team rest',
  rest_days_away: 'Away team rest',
  rest_days_gap: 'Rest gap',
  imp_prob_h: 'Market home win prob.',
  imp_prob_d: 'Market draw prob.',
  imp_prob_a: 'Market away win prob.',
}

// Each preset fills the form so you can see how the model reacts to a
// different kind of matchup without typing anything. "Newly promoted" is
// deliberately a team with zero rows in feature_state -- it exercises the
// cold-start fallback (neutral Elo/form/rest priors) documented in
// features.py and predict.py, rather than being a random extra example.
export const PRESETS = [
  {
    label: 'Title race clash',
    home: 'Arsenal',
    away: 'Man City',
    avgH: '2.60',
    avgD: '3.60',
    avgA: '2.70',
  },
  {
    label: 'Big home favorite',
    home: 'Man City',
    away: 'Sheffield United',
    avgH: '1.25',
    avgD: '6.50',
    avgA: '11.00',
  },
  {
    label: 'Relegation six-pointer',
    home: 'Luton',
    away: 'Burnley',
    avgH: '2.30',
    avgD: '3.30',
    avgA: '3.10',
  },
  {
    label: 'Newly promoted (cold start)',
    home: 'Birmingham',
    away: 'Arsenal',
    avgH: '5.50',
    avgD: '4.20',
    avgA: '1.55',
  },
]

export function confidenceTier(probabilities) {
  const top = Math.max(...Object.values(probabilities))
  if (top >= 0.55) return { label: 'High confidence', tone: 'high' }
  if (top >= 0.4) return { label: 'Moderate confidence', tone: 'moderate' }
  return { label: 'Low confidence', tone: 'low' }
}
