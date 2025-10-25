import { useState } from 'react'

function App() {
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState('')

  const handleAsk = async () => {
    const apiBase = import.meta.env.VITE_API_BASE || ''
    const res = await fetch(`/api/answer`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: question })
    })
    const data = await res.json()
    setAnswer(data.answer || 'No answer')
  }

  return (
    <div style={{ padding: '2rem' }}>
      <h1>Explaina</h1>
      <input 
        value={question}
        onChange={e => setQuestion(e.target.value)}
        placeholder="Ask anything..."
        style={{ width: '300px', padding: '8px' }}
      />
      <button onClick={handleAsk} style={{ marginLeft: '8px', padding: '8px 16px' }}>
        Ask
      </button>
      {answer && <p style={{ marginTop: '1rem' }}>{answer}</p>}
    </div>
  )
}

export default App
