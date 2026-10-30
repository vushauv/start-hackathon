import { useState } from 'react'
import QuestionList from './components/QuestionList.jsx'
import QuestionDetail from './components/QuestionDetail.jsx'

export default function App() {
  const [selected, setSelected] = useState(null)
  return (
    <div className="app">
      <aside>
        <h1>Memory Benchmark Explorer</h1>
        <QuestionList selectedId={selected?.question.question_id} onSelect={setSelected} />
      </aside>
      <main>
        {selected ? (
          <QuestionDetail key={selected.question.question_id} {...selected} />
        ) : (
          <p className="muted pad">Select a question to see its gold answer and conversation history.</p>
        )}
      </main>
    </div>
  )
}
