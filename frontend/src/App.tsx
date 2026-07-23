import { useState, type FormEvent, type CSSProperties } from 'react'

// Point this at your deployed FastAPI backend (Render URL, or localhost while
// developing). Set VITE_API_BASE_URL in a .env file to override the default.
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

interface PredictionPoint {
  id: string
  md: number | null
  pred: number
}

interface PredictResult {
  well: string
  n_points: number
  predictions: PredictionPoint[]
}

interface ApiErrorBody {
  detail?: string
}

export default function App() {
  const [file, setFile] = useState<File | null>(null)
  const [typeWellFile, setTypeWellFile] = useState<File | null>(null)
  const [wellId, setWellId] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<PredictResult | null>(null)

  async function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault()
    if (!file) {
      setError('Please choose a horizontal well CSV file first.')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    const formData = new FormData()
    formData.append('file', file)
    if (wellId.trim()) formData.append('well_id', wellId.trim())
    if (typeWellFile) formData.append('type_well_file', typeWellFile)

    try {
      const res = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        body: formData,
      })
      const data: PredictResult | ApiErrorBody = await res.json()
      if (!res.ok) {
        throw new Error((data as ApiErrorBody).detail || `Request failed with status ${res.status}`)
      }
      setResult(data as PredictResult)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong.')
    } finally {
      setLoading(false)
    }
  }

  function handleDownload() {
    if (!result) return
    const header = 'id,md,pred\n'
    const rows = result.predictions
      .map((p) => `${p.id},${p.md ?? ''},${p.pred}`)
      .join('\n')
    const blob = new Blob([header + rows], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${result.well}_predictions.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div style={{ maxWidth: 720, margin: '40px auto', padding: '0 20px', fontFamily: 'system-ui, sans-serif' }}>
      <h1 style={{ fontSize: 22, marginBottom: 4 }}>Wellbore TVT Predictor</h1>
      <p style={{ color: '#555', marginTop: 0, marginBottom: 24 }}>
        Upload a horizontal well CSV to get TVT predictions.
      </p>

      <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          Horizontal well CSV (required)
          <input
            type="file"
            accept=".csv"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
        </label>

        <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          Type well CSV (optional — uses server default if omitted)
          <input
            type="file"
            accept=".csv"
            onChange={(e) => setTypeWellFile(e.target.files?.[0] ?? null)}
          />
        </label>

        <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          Well ID (optional — derived from filename if omitted)
          <input
            type="text"
            value={wellId}
            onChange={(e) => setWellId(e.target.value)}
            placeholder="e.g. well123"
            style={{ padding: 6 }}
          />
        </label>

        <button type="submit" disabled={loading} style={{ padding: '8px 16px', width: 160 }}>
          {loading ? 'Predicting…' : 'Predict'}
        </button>
      </form>

      {error && (
        <p style={{ color: '#b00020', marginTop: 20 }}>
          Error: {error}
        </p>
      )}

      {result && (
        <div style={{ marginTop: 28 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h2 style={{ fontSize: 18, margin: 0 }}>
              Results — {result.well} ({result.n_points} points)
            </h2>
            <button onClick={handleDownload} style={{ padding: '6px 12px' }}>
              Download CSV
            </button>
          </div>

          <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: 12 }}>
            <thead>
              <tr>
                <th style={thStyle}>id</th>
                <th style={thStyle}>MD</th>
                <th style={thStyle}>Predicted TVT</th>
              </tr>
            </thead>
            <tbody>
              {result.predictions.map((p) => (
                <tr key={p.id}>
                  <td style={tdStyle}>{p.id}</td>
                  <td style={tdStyle}>{p.md ?? '—'}</td>
                  <td style={tdStyle}>{p.pred.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

const thStyle: CSSProperties = { textAlign: 'left', borderBottom: '2px solid #ccc', padding: '6px 8px', fontSize: 13 }
const tdStyle: CSSProperties = { borderBottom: '1px solid #eee', padding: '6px 8px', fontSize: 13 }
