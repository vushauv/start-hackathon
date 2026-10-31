import { useEffect, useState } from 'react'
import Explorer from './explore/Explorer.jsx'
import ChatApp from './chat/ChatApp.jsx'
import { parseHash } from './lib/route.js'

export default function App() {
  const [route, setRoute] = useState(() => parseHash(location.hash))
  useEffect(() => {
    const on = () => setRoute(parseHash(location.hash))
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])
  return route.page === 'explore' ? <Explorer /> : <ChatApp route={route} />
}
