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
// different kind of matchup without typing anything. All four teams used
// here are real, current Premier League teams (same set as CURRENT_PL_TEAMS
// in src/fixtures.py) -- no Championship sides or made-up placeholder names.
// "Newly promoted" uses Coventry specifically because it's a genuine
// current PL team with zero rows in feature_state (confirmed against the
// trained model), so it exercises the real cold-start fallback (neutral
// Elo/form/rest priors, documented in features.py/predict.py) without
// resorting to a team that doesn't actually exist in the league.
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
    away: 'Sunderland',
    avgH: '1.25',
    avgD: '6.50',
    avgA: '11.00',
  },
  {
    label: 'Relegation six-pointer',
    home: 'Hull',
    away: 'Leeds',
    avgH: '2.30',
    avgD: '3.30',
    avgA: '3.10',
  },
  {
    label: 'Newly promoted (cold start)',
    home: 'Coventry',
    away: 'Arsenal',
    avgH: '5.50',
    avgD: '4.20',
    avgA: '1.55',
  },
]

// Primary/secondary kit colors, keyed to match our team names exactly
// (football-data.co.uk style: "Man City" not "Manchester City FC", etc.).
// Hand-compiled against club crest/kit identity, cross-checked against a
// public team-colors dataset rather than typed from memory alone -- that
// source turned out to have real gaps (about a third of our 34 teams
// aren't in it at all, since it's an old EPL season snapshot) and at least
// one clear error (Liverpool listed teal-before-red), so it's a reference,
// not something copied through unverified. A team missing here (e.g. a
// one-off preset opponent that was never really promoted) falls back to a
// neutral gray swatch in TeamSwatch rather than breaking.
export const TEAM_COLORS = {
  Arsenal: ['#EF0107', '#023474'],
  'Aston Villa': ['#670E36', '#95BFE5'],
  Bournemouth: ['#DA291C', '#000000'],
  Brentford: ['#E30613', '#000000'],
  Brighton: ['#0057B8', '#FFCD00'],
  Burnley: ['#6C1D45', '#99D6EA'],
  Cardiff: ['#0070B5', '#FFC20E'],
  Chelsea: ['#034694', '#DBA111'],
  'Crystal Palace': ['#1B458F', '#C4122E'],
  Everton: ['#003399', '#FFFFFF'],
  Fulham: ['#FFFFFF', '#000000'],
  Huddersfield: ['#0E63AD', '#FFC20E'],
  Hull: ['#F18A01', '#000000'],
  Ipswich: ['#0044A9', '#FFFFFF'],
  Leeds: ['#FFFFFF', '#1D428A'],
  Leicester: ['#003090', '#FDBE11'],
  Liverpool: ['#C8102E', '#F6EB61'],
  Luton: ['#F78F1E', '#002D62'],
  'Man City': ['#6CABDD', '#1C2C5B'],
  'Man United': ['#DA020E', '#FBE122'],
  Middlesbrough: ['#E2231A', '#FFFFFF'],
  Newcastle: ['#241F20', '#FFFFFF'],
  Norwich: ['#FFF200', '#00A650'],
  "Nott'm Forest": ['#DD0000', '#FFFFFF'],
  'Sheffield United': ['#EE2737', '#FFFFFF'],
  Southampton: ['#D71920', '#FFFFFF'],
  Stoke: ['#E03A3E', '#1B449C'],
  Sunderland: ['#EB172B', '#000000'],
  Swansea: ['#FFFFFF', '#000000'],
  Tottenham: ['#132257', '#FFFFFF'],
  Watford: ['#FBEE23', '#ED2127'],
  'West Brom': ['#122F67', '#FFFFFF'],
  'West Ham': ['#7A263A', '#1BB1E7'],
  Wolves: ['#FDB913', '#231F20'],
}

export function confidenceTier(probabilities) {
  const top = Math.max(...Object.values(probabilities))
  if (top >= 0.55) return { label: 'High confidence', tone: 'high' }
  if (top >= 0.4) return { label: 'Moderate confidence', tone: 'moderate' }
  return { label: 'Low confidence', tone: 'low' }
}
