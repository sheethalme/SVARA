import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './styles.css'

const LANG_NAMES = { en: 'English', hi: 'हिन्दी', ml: 'മലയാളം', kn: 'ಕನ್ನಡ' }
// the server substitutes every literal token placeholder in the page, so the sentinel is built at runtime
const PLACEHOLDER = '__TOK' + 'EN__'
const TOKEN = document.querySelector('meta[name="call-token"]')?.content || PLACEHOLDER
const HAS_BACKEND_TOKEN = TOKEN !== PLACEHOLDER

const FALLBACK_FORMS = [
  {
    id: 'clinic_registration', title: 'Clinic registration', language: 'en', languages: ['en', 'hi', 'ml', 'kn'],
    fields: [
      { name: 'name', label: 'Full name', required: true }, { name: 'age', label: 'Age', required: true },
      { name: 'gender', label: 'Gender', required: true }, { name: 'phone', label: 'Mobile number', required: true },
      { name: 'blood_group', label: 'Blood group', required: false },
    ], accent: 'green', eyebrow: 'Healthcare', description: 'A calm intake for first-time clinic visits, built for voice from the first question to the final confirmation.',
  },
  {
    id: 'greenleaf_multispeciality_hospital', title: 'Medical form', language: 'en', languages: ['en'],
    fields: [
      { name: 'name', label: 'Patient name', required: true }, { name: 'date_of_birth', label: 'Date of birth', required: true },
      { name: 'age', label: 'Age', required: true }, { name: 'phone', label: 'Mobile number', required: true },
      { name: 'blood_group', label: 'Blood group', required: true }, { name: 'contact_person_name', label: 'Contact person name', required: true },
      { name: 'relationship', label: 'Relationship', required: true }, { name: 'contact_phone', label: 'Contact phone', required: true },
      { name: 'i_agree', label: 'I agree', required: true },
    ], accent: 'burgundy', eyebrow: 'Specialist care', description: 'A richer patient and emergency-contact intake for multi-speciality care teams.',
  },
  {
    id: 'sunrise_institute_of_technology', title: 'Sunrise Institute of Technology', language: 'en', languages: ['en'],
    fields: Array.from({ length: 18 }, (_, index) => ({ name: `field_${index}`, label: ['Father\'s name', 'Mother\'s name', 'Date of birth', 'Age', 'Gender', 'Nationality', 'Mobile number', 'Email ID', 'Permanent address', 'PIN code', 'Aadhaar number', 'Category', 'Last school / college', 'Board / university', 'Year of passing', 'Percentage / CGPA', 'Branch preferred', 'Place'][index], required: index !== 16 })),
    accent: 'bone', eyebrow: 'Admissions', description: 'A long-form admission flow that stays human when the paperwork gets complicated.',
  },
]

const FALLBACK_CALLS = [
  { id: 'a84f2b91', form_title: 'Clinic registration', outcome: 'completed', language: 'en', started_at: '2026-11-23T09:42:00Z', duration: '02:14', fields_completed: 5 },
  { id: 'd1309c54', form_title: 'Medical form', outcome: 'in_progress', language: 'en', started_at: '2026-11-23T08:26:00Z', duration: '01:08', fields_completed: 4 },
  { id: 'f029a1cc', form_title: 'Clinic registration', outcome: 'completed', language: 'ml', started_at: '2026-11-22T15:18:00Z', duration: '03:02', fields_completed: 5 },
  { id: 'c73384ed', form_title: 'Sunrise Institute of Technology', outcome: 'needs_help', language: 'en', started_at: '2026-11-22T11:06:00Z', duration: '04:37', fields_completed: 11 },
]

const DEMO_TURNS = [
  { say: 'Hello. I will ask a few questions to fill in your Clinic registration form. Do you agree?', heard: 'Yes, I agree.', mode: 'yesno' },
  { say: 'What is your full name?', heard: 'Anil Kumar', mode: 'answer' },
  { say: 'I heard your name as Anil Kumar. Is that correct?', heard: 'Yes', mode: 'yesno' },
  { say: 'How old are you?', heard: 'Twenty five', mode: 'answer' },
  { say: 'I heard your age as 25. Is that correct?', heard: 'Yes', mode: 'yesno' },
  { say: 'What is your mobile number? Please say the digits one by one.', heard: 'Nine eight seven six five four three two one zero', mode: 'answer' },
  { say: 'Thank you. Your form has been saved. Goodbye.', heard: '', mode: 'done', done: true },
]

function Icon({ name, size = 18, stroke = 1.8 }) {
  const paths = {
    home: <><path d="m3 10 9-7 9 7"/><path d="M5 9v11h14V9"/><path d="M9 20v-6h6v6"/></>,
    grid: <><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></>,
    pulse: <><path d="M3 12h4l2.2-6 4.2 12 2.2-6H21"/></>,
    settings: <><path d="M12 15.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7Z"/><path d="m19.4 15 .1.1a2 2 0 0 1-2.8 2.8l-.1-.1a2 2 0 0 0-3.4 1.4v.2a2 2 0 0 1-4 0v-.2a2 2 0 0 0-3.4-1.4l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1A2 2 0 0 0 3.7 12a2 2 0 0 0-1.7-2v-.1a2 2 0 0 1 2-2h.2a2 2 0 0 0 1.4-3.4l-.1-.1a2 2 0 0 1 2.8-2.8l.1.1A2 2 0 0 0 12 3.7 2 2 0 0 0 14 2v-.1a2 2 0 0 1 2 2v.2a2 2 0 0 0 3.4 1.4l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1A2 2 0 0 0 20.3 12a2 2 0 0 0 1.7 2v.1a2 2 0 0 1-2 2h-.2a2 2 0 0 0-.4-1.1Z" transform="scale(.78) translate(3.3 3.3)"/></>,
    arrow: <><path d="M5 12h14"/><path d="m13 6 6 6-6 6"/></>,
    plus: <><path d="M12 5v14M5 12h14"/></>,
    mic: <><rect x="8" y="3" width="8" height="12" rx="4"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3M8 21h8"/></>,
    wave: <><path d="M3 12h2l2-5 3 10 3-14 3 14 2-5h3"/></>,
    check: <path d="m5 12 4 4L19 6"/>,
    clock: <><circle cx="12" cy="12" r="8.5"/><path d="M12 7v5l3 2"/></>,
    chevron: <path d="m9 6 6 6-6 6"/>,
    close: <><path d="m6 6 12 12M18 6 6 18"/></>,
    spark: <><path d="m12 3 1.6 5.4L19 10l-5.4 1.6L12 17l-1.6-5.4L5 10l5.4-1.6L12 3Z"/><path d="m19 16 .7 2.3L22 19l-2.3.7L19 22l-.7-2.3L16 19l2.3-.7L19 16Z"/></>,
    download: <><path d="M12 3v12M7 10l5 5 5-5M5 21h14"/></>,
    lock: <><rect x="5" y="11" width="14" height="9" rx="2"/><path d="M8 11V8a4 4 0 0 1 8 0v3"/></>,
    shield: <><path d="M12 3 19 6v5c0 4.5-2.9 8.2-7 10-4.1-1.8-7-5.5-7-10V6l7-3Z"/><path d="m9 12 2 2 4-4"/></>,
  }
  return <svg aria-hidden="true" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={stroke} strokeLinecap="round" strokeLinejoin="round">{paths[name] || paths.spark}</svg>
}

const store = {
  get: key => { try { return window.localStorage.getItem(key) } catch { return null } },
  set: (key, value) => { try { window.localStorage.setItem(key, value) } catch { /* storage unavailable */ } },
}

const ACCENTS = ['green', 'burgundy', 'bone']
function decorateForms(list, previous = []) {
  return list.map((form, index) => {
    const old = previous.find(item => item.id === form.id)
    return { accent: old?.accent || ACCENTS[index % ACCENTS.length], eyebrow: old?.eyebrow || 'Voice form', description: old?.description || `A hands-free voice flow with ${form.fields?.length || 0} questions.`, ...form }
  })
}

function normalizeCall(call) {
  const started = call.started_at && call.ended_at ? Math.max(0, Math.round((new Date(call.ended_at) - new Date(call.started_at)) / 1000)) : null
  return { ...call, id: call.id || call.call_id || String(Math.random()).slice(2, 10), duration: call.duration || (started == null || Number.isNaN(started) ? '—' : `${String(Math.floor(started / 60)).padStart(2, '0')}:${String(started % 60).padStart(2, '0')}`), fields_completed: call.fields_completed ?? call.progress?.[0] }
}

class ErrorBoundary extends React.Component {
  state = { error: null }
  static getDerivedStateFromError(error) { return { error } }
  render() {
    if (!this.state.error) return this.props.children
    return <div style={{ padding: 32, maxWidth: 640, margin: '10vh auto', fontFamily: 'system-ui' }}><h2>SVARA hit a snag</h2><p>{String(this.state.error?.message || this.state.error)}</p><button className="button button--primary" onClick={() => { window.location.href = '/' }}>Reload SVARA</button></div>
  }
}

function formatDate(value) {
  if (!value || Number.isNaN(new Date(value).getTime())) return '—'
  return new Intl.DateTimeFormat('en-IN', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' }).format(new Date(value))
}

function statusLabel(outcome) {
  return { completed: 'Saved', in_progress: 'In progress', needs_help: 'Needs attention', hangup: 'Interrupted', declined: 'Declined' }[outcome] || outcome || 'Unknown'
}

let authToken = ''
let onSessionExpired = null

async function api(path, options = {}) {
  const headers = { ...(options.headers || {}) }
  if (HAS_BACKEND_TOKEN) headers['X-Call-Token'] = TOKEN
  if (authToken) headers['X-Session'] = authToken
  const response = await fetch(path, { ...options, headers })
  if (!response.ok) {
    let info = {}
    try { info = await response.json() } catch { /* response was not JSON */ }
    const error = new Error(info.stale_page ? 'SVARA was restarted since this page opened. Reload the page (Ctrl+F5) and try again.' : (info.error || response.statusText || 'Request failed'))
    error.status = response.status
    error.info = info
    if (response.status === 401 && info.session_expired) onSessionExpired?.()
    throw error
  }
  return response
}

async function authCall(action, body) {
  if (!HAS_BACKEND_TOKEN) throw new Error('The SVARA server is not connected. Start it with: python -m callpage')
  return jsonApi(`/api/auth/${action}`, body)
}

async function downloadFilled(callId, title) {
  const blob = await (await api(`/api/call/${callId}/filled.pdf`)).blob()
  const link = document.createElement('a'); link.href = URL.createObjectURL(blob); link.download = `${title || 'form'}-filled.pdf`; link.click(); URL.revokeObjectURL(link.href)
}

async function jsonApi(path, body) {
  const response = await api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
  return response.json()
}


function useTilt(max = 8) {
  return {
    onPointerMove: event => {
      const b = event.currentTarget.getBoundingClientRect()
      event.currentTarget.style.setProperty('--tx', `${((event.clientY - b.top) / b.height - 0.5) * -max}deg`)
      event.currentTarget.style.setProperty('--ty', `${((event.clientX - b.left) / b.width - 0.5) * max}deg`)
    },
    onPointerLeave: event => { event.currentTarget.style.setProperty('--tx', '0deg'); event.currentTarget.style.setProperty('--ty', '0deg') },
  }
}

// A draggable 3D ring of audio bars with floating cubes; drag to spin, it eases back to a slow idle rotation.
function SignalScene({ amp = 0.5, children, size = 340 }) {
  const stage = useRef(null)
  const state = useRef({ rx: -58, ry: 0, vy: 0.25, vx: 0, drag: false, x: 0, y: 0, scrollY: window.scrollY })
  useEffect(() => {
    let frame
    const loop = () => {
      const st = state.current
      st.ry += (window.scrollY - st.scrollY) * 0.35; st.scrollY = window.scrollY
      if (!st.drag && document.documentElement.dataset.motion !== 'off') { st.ry += st.vy; st.rx += (-58 - st.rx) * 0.02; st.vy += (0.25 - st.vy) * 0.02 }
      if (stage.current) stage.current.style.transform = `rotateX(${st.rx}deg) rotateY(${st.ry}deg)`
      frame = requestAnimationFrame(loop)
    }
    loop()
    return () => cancelAnimationFrame(frame)
  }, [])
  const down = event => { const st = state.current; st.drag = true; st.x = event.clientX; st.y = event.clientY; event.currentTarget.setPointerCapture(event.pointerId) }
  const move = event => {
    const st = state.current
    if (!st.drag) return
    st.vy = (event.clientX - st.x) * 0.4; st.ry += event.clientX - st.x; st.rx = Math.max(-85, Math.min(-20, st.rx - (event.clientY - st.y) * 0.4)); st.x = event.clientX; st.y = event.clientY
  }
  const up = () => { state.current.drag = false }
  const bars = 36
  return <div className="scene3d" style={{ '--scene': `${size}px`, '--amp': amp }} onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerCancel={up} aria-hidden="true">
    <div className="scene3d__stage" ref={stage}>
      {Array.from({ length: bars }, (_, i) => <i key={i} className="scene3d__bar" style={{ '--a': `${(360 / bars) * i}deg`, '--d': `${(i % 9) * -0.18}s`, '--h': 18 + ((i * 37) % 30) }} />)}
      {[0, 1, 2].map(i => <b key={i} className={`scene3d__cube scene3d__cube--${i}`}><span /><span /><span /><span /><span /><span /></b>)}
    </div>
    <div className="scene3d__center">{children}</div>
  </div>
}

function Orb({ compact = false, active = false }) {
  const handlePointerMove = event => {
    const bounds = event.currentTarget.getBoundingClientRect()
    const x = ((event.clientX - bounds.left) / bounds.width - 0.5) * 18
    const y = ((event.clientY - bounds.top) / bounds.height - 0.5) * -18
    event.currentTarget.style.setProperty('--orb-tilt-x', `${x}deg`)
    event.currentTarget.style.setProperty('--orb-tilt-y', `${y}deg`)
  }
  const resetPointer = event => {
    event.currentTarget.style.setProperty('--orb-tilt-x', '0deg')
    event.currentTarget.style.setProperty('--orb-tilt-y', '0deg')
  }
  return <div className={`orb ${compact ? 'orb--compact' : ''} ${active ? 'orb--active' : ''}`} onPointerMove={handlePointerMove} onPointerLeave={resetPointer} aria-hidden="true">
    <div className="orb-halo" /><div className="orb-grid" /><div className="orb-scene">
      <div className="orb-ring orb-ring--one" /><div className="orb-ring orb-ring--two" />
      <div className="orb-prism"><span /><span /><span /><span /><span /><span /></div>
      <div className="orb-core"><span /><span /><span /></div><div className="orb-spark orb-spark--one" /><div className="orb-spark orb-spark--two" />
    </div>
  </div>
}

function BrandMark() {
  return <span className="brand-mark brand-mark--svg"><img src="/svara-mark.svg" alt="" /></span>
}

function initials(name = '') { return name.split(/\s+/).filter(Boolean).slice(0, 2).map(part => part[0].toUpperCase()).join('') || 'U' }
function greeting(name) { const hour = new Date().getHours(); return `Good ${hour < 12 ? 'morning' : hour < 18 ? 'afternoon' : 'evening'}, ${(name || 'there').split(' ')[0]}` }
const navItems = role => role === 'admin'
  ? [['overview', 'Dashboard', 'home'], ['forms', 'Form studio', 'grid'], ['responses', 'Responses', 'pulse']]
  : [['overview', 'Home', 'home'], ['forms', 'Choose a form', 'grid']]

function Sidebar({ session, view, setView, onLogout }) {
  const role = session.role
  return <aside className="sidebar">
    <div className="brand"><BrandMark /><div><strong>SVARA</strong><span>voice-first forms</span></div></div>
    <div className={`role-badge role-badge--${role}`}><span className="workspace-dot" /><span>{role === 'admin' ? 'Admin console' : 'Caller space'}</span></div>
    <nav className="nav" aria-label="Primary navigation">
      <span className="nav-label">Workspace</span>
      {navItems(role).map(([id, label, icon]) => <button key={id} className={`nav-item ${view === id ? 'is-active' : ''}`} onClick={() => setView(id)}><Icon name={icon} /><span>{label}</span></button>)}
    </nav>
    <div className="sidebar-bottom">
      {role === 'admin' && <button className={`nav-item ${view === 'settings' ? 'is-active' : ''}`} onClick={() => setView('settings')}><Icon name="settings" /><span>Settings</span></button>}
      <div className="sidebar-tip"><div className="tip-icon"><Icon name="spark" size={16} /></div><div><strong>Voice-first by design</strong><p>One tap, then let the conversation flow.</p></div></div>
      <div className="profile"><div className="avatar">{initials(session.name)}</div><div><strong>{session.name}</strong><span>{role === 'admin' ? 'Administrator' : 'Caller'}</span></div><button aria-label="Sign out" title="Sign out" onClick={onLogout}><Icon name="close" size={15} /></button></div>
    </div>
  </aside>
}

function MobileNav({ role, view, setView }) {
  return <nav className="mobile-nav" aria-label="Primary navigation">{[...navItems(role), ...(role === 'admin' ? [['settings', 'Settings', 'settings']] : [])].map(([id, label, icon]) => <button key={id} className={view === id ? 'is-active' : ''} onClick={() => setView(id)}><Icon name={icon} size={20} /><span>{label}</span></button>)}</nav>
}

function Topbar({ role, view, onStart, backendConnected, name, onLogout, onUpload }) {
  const titles = role === 'admin'
    ? { overview: [greeting(name), 'Shape the conversation, then let SVARA carry it.'], forms: ['Form studio', 'Create, edit, and publish the flows your callers need.'], responses: ['Response history', 'Every conversation, captured with care.'], settings: ['Settings', 'Your account, the speech engine and how each form handles privacy.'] }
    : { overview: [greeting(name), 'Choose a form and let SVARA do the asking.'], forms: ['Choose a form', 'A calmer way to share the details that matter.'] }
  const [title, sub] = titles[view] || titles.overview
  return <header className="topbar"><div><div className="breadcrumb"><span>{role === 'admin' ? 'ADMIN CONSOLE' : 'CALLER EXPERIENCE'}</span><span>/</span><strong>{view}</strong></div><h1>{title}</h1><p>{sub}</p></div><div className="topbar-actions"><div className={`connection-status ${backendConnected ? 'is-connected' : ''}`}><span />{backendConnected ? 'Backend connected' : 'Demo workspace'}</div><button className="text-button signout-btn" onClick={onLogout}>Sign out</button><button className="button button--outline button--small topbar-upload" onClick={onUpload}><Icon name="download" size={15} />{role === 'admin' ? 'Upload form' : 'Upload my form'}</button><button className="button button--primary button--small" onClick={onStart}><Icon name="plus" size={16} />{role === 'admin' ? 'Start test call' : 'Start voice call'}</button></div></header>
}

function StatCard({ label, value, meta, tone = 'green', icon }) {
  const tilt = useTilt(10)
  return <div className={`stat-card stat-card--${tone}`} {...tilt}><div className="stat-card__top"><span>{label}</span><span className="stat-icon"><Icon name={icon} size={17} /></span></div><strong>{value}</strong><span className="stat-card__meta">{meta}</span></div>
}

function Overview({ forms, calls, onStart, setView, role = 'admin', onEdit, onDelete, onAccess, onUnlockCode }) {
  const completed = calls.filter(call => call.outcome === 'completed').length
  const admin = role === 'admin'
  return <div className="view-stack">
    <section className="hero-card">
      <div className="hero-copy"><div className="eyebrow"><span className="eyebrow-line" />VOICE-FIRST INTAKE</div><h2>Forms that <em>listen</em> back.</h2><p>SVARA turns long, tiring forms into a guided conversation — warm, hands-free, and easy to trust.</p><div className="hero-actions"><button className="button button--bone" onClick={onStart}>Start a voice call <Icon name="arrow" size={17} /></button><button className="text-button" onClick={() => setView('forms')}>Browse forms <Icon name="arrow" size={15} /></button></div><div className="hero-note"><span className="note-check"><Icon name="check" size={13} /></span><span>No typing during the call</span><span className="note-separator" /><span>4 languages ready</span></div></div>
      <div className="hero-visual"><div className="visual-caption visual-caption--top"><span className="pulse-dot" />Live voice engine</div><SignalScene><Orb compact /></SignalScene><div className="visual-caption visual-caption--bottom"><span>Natural pauses</span><span className="mini-wave"><i /><i /><i /><i /><i /><i /><i /></span></div></div>
    </section>
    {!admin && <AccessCodeCard onUnlock={onUnlockCode} onStart={onStart} />}
    {admin ? (
    <section className="stats-grid"><StatCard label="Forms ready" value={forms.length} meta="Across 3 workflows" tone="green" icon="grid" /><StatCard label="Calls this month" value="128" meta="+18.4% vs last month" tone="bone" icon="pulse" /><StatCard label="Saved responses" value={completed ? '92%' : '92%'} meta="Completion rate" tone="burgundy" icon="check" /><StatCard label="Avg. call time" value="02:14" meta="12 sec faster this week" tone="green" icon="clock" /></section>
    ) : (
    <section className="steps-grid">{[['1', 'Choose a form', 'Pick the form you need and your language.', 'grid'], ['2', 'Allow the microphone', 'Your browser asks once. A headset works best.', 'mic'], ['3', 'Just talk', 'SVARA asks, listens and confirms. No typing.', 'wave'], ['4', 'All saved', 'Your answers are stored when the call ends.', 'check']].map(([n, t, d, i]) => <div className="step-card" key={n}><span className="step-card__n">{n}</span><span className="step-card__icon"><Icon name={i} size={18} /></span><strong>{t}</strong><p>{d}</p></div>)}</section>
    )}
    <section className="section-heading"><div><span className="section-kicker">YOUR FLOWS</span><h3>Pick up where you left off</h3></div><button className="text-button" onClick={() => setView('forms')}>View all forms <Icon name="arrow" size={15} /></button></section>
    {!admin && forms.length === 0 && <EmptyForms />}<div className="form-grid">{forms.slice(0, 3).map(form => <FormCard key={form.id} form={form} role={role} onEdit={onEdit} onDelete={onDelete} onAccess={onAccess} onStart={onStart} compact />)}</div>
    {admin && <>
    <section className="section-heading section-heading--recent"><div><span className="section-kicker">RECENT ACTIVITY</span><h3>Conversations with momentum</h3></div><button className="text-button" onClick={() => setView(admin ? 'responses' : 'forms')}>{admin ? 'See all responses' : 'Choose another form'} <Icon name="arrow" size={15} /></button></section>
    <RecentCalls calls={calls.slice(0, 3)} />
    </>}
  </div>
}

function downloadForm(form) {
  const data = { id: form.id, title: form.title, language: form.language || 'en', languages: form.languages || ['en'], settings: form.settings || {},
    fields: (form.fields || []).map(field => ({ name: field.name, label: field.label, kind: field.kind || 'text', required: Boolean(field.required), ...(field.question ? { question: field.question } : {}) })) }
  const link = document.createElement('a')
  link.href = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }))
  link.download = `${form.id}.json`; link.click(); URL.revokeObjectURL(link.href)
}

function CardMenu({ form, onEdit, onDelete, onAccess }) {
  const [open, setOpen] = useState(false)
  const box = useRef(null)
  useEffect(() => {
    if (!open) return undefined
    const close = event => { if (event.type === 'keydown' ? event.key === 'Escape' : !box.current?.contains(event.target)) setOpen(false) }
    document.addEventListener('pointerdown', close); document.addEventListener('keydown', close)
    return () => { document.removeEventListener('pointerdown', close); document.removeEventListener('keydown', close) }
  }, [open])
  const run = action => () => { setOpen(false); action() }
  return <div className="card-menu-wrap" ref={box}>
    <button className="form-card__more form-card__more--button" aria-haspopup="menu" aria-expanded={open} aria-label={`Options for ${form.title}`} onClick={() => setOpen(value => !value)}>•••</button>
    {open && <div className="card-menu" role="menu">
      {onEdit && <button role="menuitem" onClick={run(() => onEdit(form))}><Icon name="settings" size={15} />Edit form</button>}
      {onAccess && <button role="menuitem" onClick={run(() => onAccess(form))}><Icon name="lock" size={15} />Access code</button>}
      <button role="menuitem" onClick={run(() => downloadForm(form))}><Icon name="download" size={15} />Save a copy (.json)</button>
      <button role="menuitem" className="is-danger" onClick={run(() => onDelete?.(form))}><Icon name="close" size={15} />Delete form</button>
    </div>}
  </div>
}

function FormCard({ form, onStart, onEdit, onDelete, onAccess, role = 'user', compact = false }) {
  const isBone = form.accent === 'bone'
  const handlePointerMove = event => {
    const bounds = event.currentTarget.getBoundingClientRect()
    const x = ((event.clientX - bounds.left) / bounds.width - 0.5) * 9
    const y = ((event.clientY - bounds.top) / bounds.height - 0.5) * -9
    event.currentTarget.style.setProperty('--tilt-x', `${y}deg`)
    event.currentTarget.style.setProperty('--tilt-y', `${x}deg`)
    event.currentTarget.style.setProperty('--spot-x', `${event.clientX - bounds.left}px`)
    event.currentTarget.style.setProperty('--spot-y', `${event.clientY - bounds.top}px`)
  }
  const resetPointer = event => { event.currentTarget.style.setProperty('--tilt-x', '0deg'); event.currentTarget.style.setProperty('--tilt-y', '0deg') }
  return <article className={`form-card form-card--${form.accent} ${compact ? 'form-card--compact' : ''}`} onPointerMove={handlePointerMove} onPointerLeave={resetPointer}><div className="form-card__spotlight" /><div className="form-card__glow" /><div className="form-card__top"><span className="form-chip"><span />{form.draft ? 'Draft · hidden from callers' : form.mine ? 'Your private form' : form.eyebrow}</span>{role === 'admin' || form.mine ? <CardMenu form={form} onEdit={role === 'admin' ? onEdit : undefined} onDelete={onDelete} onAccess={role === 'admin' && !form.draft ? onAccess : undefined} /> : <span className="form-card__more">•••</span>}</div><h4>{form.title}</h4><p>{form.description}</p><div className="form-card__meta"><span><Icon name="grid" size={14} />{form.field_count ?? form.fields.length} {(form.field_count ?? form.fields.length) === 1 ? 'field' : 'fields'}</span><span><Icon name="wave" size={14} />{form.languages.length} {form.languages.length === 1 ? 'language' : 'languages'}</span>{role === 'admin' && !form.draft && (form.access_code ? <button className="code-chip" title="Manage the access code" onClick={() => onAccess?.(form)}><Icon name="lock" size={12} />{form.access_code}</button> : form.has_code === false ? <button className="code-chip code-chip--open" title="Set an access code" onClick={() => onAccess?.(form)}>Open to all</button> : null)}</div><div className="form-card__footer"><div className="language-stack">{form.languages.slice(0, 4).map(lang => <span key={lang} title={LANG_NAMES[lang]}>{lang.toUpperCase()}</span>)}</div><div className="form-card__actions">{role === 'admin' && <button className="card-edit-link" onClick={() => onEdit?.(form)}>Edit</button>}<button className={`icon-button ${isBone ? 'icon-button--dark' : ''}`} onClick={() => onStart(form)} aria-label={`${role === 'admin' ? 'Test' : 'Start'} ${form.title} call`}><Icon name="arrow" size={17} /></button></div></div></article>
}

function RecentCalls({ calls, onSelect }) {
  return <div className="recent-list">{calls.map(call => <button className="recent-row" key={call.id} onClick={() => onSelect?.(call)}><div className="recent-avatar"><Icon name={call.outcome === 'completed' ? 'check' : 'pulse'} size={16} /></div><div className="recent-main"><strong>{call.form_title}</strong><span>{formatDate(call.started_at)} · {LANG_NAMES[call.language] || call.language}</span></div><div className="recent-status"><span className={`status-dot status-dot--${call.outcome}`} />{statusLabel(call.outcome)}</div><span className="recent-duration">{call.duration || '—'}</span><Icon name="chevron" size={16} /></button>)}</div>
}

function FormsView({ forms, onStart, role = 'user', onEdit, onDelete, onAccess, onCreate, onUpload, onUnlockCode }) {
  const admin = role === 'admin'
  return <div className="view-stack"><section className="forms-intro"><div><span className="section-kicker">{admin ? 'FORM STUDIO' : 'FORM LIBRARY'}</span><h2>{admin ? 'Build a better conversation.' : 'Make every question feel lighter.'}</h2><p>{admin ? 'Create a flow, tune every spoken question, and publish it when your team is ready.' : 'Each flow is built from your backend schema and can be answered naturally, in the language your caller prefers.'}</p>{!admin && <div className="intro-actions"><button className="button button--primary" onClick={onUpload}><Icon name="download" size={16} />Upload your own form</button></div>}{admin && <div className="intro-actions"><button className="button button--primary" onClick={onUpload}><Icon name="download" size={16} />Upload a PDF form</button><button className="button button--outline" onClick={onCreate}><Icon name="plus" size={16} />Create from scratch</button></div>}</div><div className="forms-intro-mark"><Orb compact /></div></section>{!admin && <AccessCodeCard onUnlock={onUnlockCode} onStart={onStart} />}<div className="library-toolbar"><span><strong>{forms.length}</strong> {admin ? 'published flows' : 'available flows'}</span><div className="filter-pills"><button className="filter-pill is-selected">All forms</button><button className="filter-pill">Healthcare</button><button className="filter-pill">Admissions</button></div></div>{!admin && forms.length === 0 && <EmptyForms />}<div className="form-grid form-grid--library">{forms.map(form => <FormCard key={form.id} form={form} role={role} onEdit={onEdit} onDelete={onDelete} onAccess={onAccess} onStart={onStart} />)}<button className="create-card create-card--upload" onClick={onUpload}><span className="create-card__icon"><Icon name="download" size={22} /></span><strong>{admin ? 'Upload a PDF form' : 'Upload your own form'}</strong><span>{admin ? 'SVARA reads the blanks and turns them into spoken questions.' : 'Bring a form you need to fill. Only you can see it, and you can download it filled in.'}</span></button>{admin && <button className="create-card" onClick={onCreate}><span className="create-card__icon"><Icon name="plus" size={22} /></span><strong>Design a new flow</strong><span>Start with a title, then add the questions your callers need.</span></button>}</div></div>
}

function ResponsesView({ calls, onSelect }) {
  const [filter, setFilter] = useState('all')
  const shown = filter === 'all' ? calls : calls.filter(call => call.outcome === filter)
  return <div className="view-stack"><section className="responses-head"><div><span className="section-kicker">VOICE LOG</span><h2>Responses, without the spreadsheet sprawl.</h2><p>A clear trail of what was asked, what was heard, and what was saved.</p></div><button className="button button--outline"><Icon name="download" size={16} />Export CSV</button></section><div className="response-summary"><div><strong>128</strong><span>Total calls</span></div><div><strong>92%</strong><span>Saved successfully</span></div><div><strong>02:14</strong><span>Average duration</span></div><div className="summary-orb"><Orb compact /></div></div><div className="table-card"><div className="table-toolbar"><div className="filter-pills"><button className={`filter-pill ${filter === 'all' ? 'is-selected' : ''}`} onClick={() => setFilter('all')}>All <span>{calls.length}</span></button><button className={`filter-pill ${filter === 'completed' ? 'is-selected' : ''}`} onClick={() => setFilter('completed')}>Saved</button><button className={`filter-pill ${filter === 'in_progress' ? 'is-selected' : ''}`} onClick={() => setFilter('in_progress')}>In progress</button><button className={`filter-pill ${filter === 'needs_help' ? 'is-selected' : ''}`} onClick={() => setFilter('needs_help')}>Needs attention</button></div><span className="table-count">Showing {shown.length} conversations</span></div><div className="table-scroll"><table><thead><tr><th>Form</th><th>Outcome</th><th>Language</th><th>Started</th><th>Duration</th><th /></tr></thead><tbody>{shown.map(call => <tr key={call.id} onClick={() => onSelect(call)}><td><div className="table-form"><span className={`table-form-icon table-form-icon--${call.outcome}`}><Icon name={call.outcome === 'completed' ? 'check' : 'pulse'} size={15} /></span><div><strong>{call.form_title}</strong><span>#{call.id}</span></div></div></td><td><span className={`pill-status pill-status--${call.outcome}`}><span />{statusLabel(call.outcome)}</span></td><td>{LANG_NAMES[call.language] || call.language}</td><td>{formatDate(call.started_at)}</td><td>{call.duration || '—'}</td><td><Icon name="chevron" size={16} /></td></tr>)}</tbody></table></div></div></div>
}

function CallSetup({ form, setForm, forms, language, setLanguage, onStart, onClose }) {
  const current = form || forms[0]
  const chooseForm = nextForm => { setForm(nextForm); setLanguage(nextForm?.language || nextForm?.languages?.[0] || 'en') }
  return <div className="modal-backdrop"><section className="setup-modal" role="dialog" aria-modal="true" aria-labelledby="setup-title"><button className="modal-close" onClick={onClose} aria-label="Close setup"><Icon name="close" size={18} /></button><div className="setup-modal__visual"><Orb compact /><span>ONE TAP<br />THEN VOICE</span></div><div className="setup-modal__content"><div className="eyebrow"><span className="eyebrow-line" />READY WHEN YOU ARE</div><h2 id="setup-title">Start a voice call</h2><p className="modal-lead">Pick a flow and language. Once the call begins, SVARA will ask, listen, and confirm without needing another tap.</p><label className="field-label" htmlFor="flow">Choose a form</label><div className="select-wrap"><select id="flow" value={current?.id || ''} onChange={event => chooseForm(forms.find(item => item.id === event.target.value))}>{forms.map(item => <option key={item.id} value={item.id}>{item.title}</option>)}</select><Icon name="chevron" size={16} /></div><label className="field-label" htmlFor="language">Conversation language</label><div className="language-options">{(current?.languages || ['en']).map(lang => <button key={lang} className={`language-option ${language === lang ? 'is-selected' : ''}`} onClick={() => setLanguage(lang)}><span>{LANG_NAMES[lang]}</span><small>{lang.toUpperCase()}</small></button>)}</div><div className="permission-note"><span><Icon name="mic" size={17} /></span><div><strong>Microphone access</strong><p>Your browser will ask once. A headset gives the clearest experience.</p></div></div><button className="button button--primary button--wide" onClick={() => onStart(current)}>Begin conversation <Icon name="arrow" size={17} /></button></div></section></div>
}

function CallOrb({ phase }) {
  return <div className={`call-orb call-orb--${phase}`}><div className="call-orb__halo" /><div className="call-orb__ring call-orb__ring--one" /><div className="call-orb__ring call-orb__ring--two" /><div className="call-orb__core"><span /><span /><span /><span /></div><div className="call-orb__label">SVARA</div></div>
}

function CallModal({ form, language, onClose, onFinished, initials: userInitials = 'U' }) {
  const [phase, setPhase] = useState('speaking')
  const [turn, setTurn] = useState({ say: 'Preparing your conversation…', heard: '' })
  const [timeline, setTimeline] = useState([])
  const [progress, setProgress] = useState([0, form?.fields?.length || 5])
  const [level, setLevel] = useState(0.2)
  const [message, setMessage] = useState('Starting secure voice session')
  const [error, setError] = useState('')
  const [demo, setDemo] = useState(!HAS_BACKEND_TOKEN)
  const [finished, setFinished] = useState(false)
  const [pdfNote, setPdfNote] = useState('')
  const timers = useRef([])
  const micRef = useRef(null)
  const demoIndex = useRef(0)
  const realCallRef = useRef(null)

  const addTimeline = useCallback((who, text) => {
    if (!text) return
    setTimeline(items => [...items, { id: `${Date.now()}-${items.length}`, who, text }])
  }, [])

  const speak = useCallback(async text => {
    if (!text) return
    setPhase('speaking'); setMessage('SVARA is speaking'); setLevel(0.08)
    if (!window.speechSynthesis || demo) { await new Promise(resolve => setTimeout(resolve, Math.min(2600, 500 + text.length * 22))); return }
    await new Promise(resolve => {
      const utterance = new SpeechSynthesisUtterance(text)
      utterance.lang = language === 'en' ? 'en-IN' : `${language}-IN`
      utterance.rate = 0.96
      utterance.onend = resolve; utterance.onerror = resolve
      window.speechSynthesis.cancel(); window.speechSynthesis.speak(utterance)
    })
  }, [demo, language])

  const encodeWav = useCallback((samples, sampleRate) => {
    const pcm = new Int16Array(samples.length)
    samples.forEach((sample, index) => { const value = Math.max(-1, Math.min(1, sample)); pcm[index] = value < 0 ? value * 32768 : value * 32767 })
    const buffer = new ArrayBuffer(44 + pcm.length * 2); const view = new DataView(buffer)
    const write = (offset, text) => [...text].forEach((char, index) => view.setUint8(offset + index, char.charCodeAt(0)))
    write(0, 'RIFF'); view.setUint32(4, 36 + pcm.length * 2, true); write(8, 'WAVEfmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true); write(36, 'data'); view.setUint32(40, pcm.length * 2, true); new Int16Array(buffer, 44).set(pcm)
    return new Blob([buffer], { type: 'audio/wav' })
  }, [])

  // One microphone for the whole call: opening it once (instead of before every answer) removes a delay per turn.
  const closeMic = useCallback(() => {
    const mic = micRef.current
    if (!mic) return
    micRef.current = null
    try { mic.node.disconnect(); mic.source.disconnect() } catch { /* already disconnected */ }
    mic.stream.getTracks().forEach(track => track.stop())
    mic.context.close().catch(() => {})
  }, [])

  const openMic = useCallback(async () => {
    if (micRef.current) return micRef.current
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('This browser cannot access a microphone.')
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } })
    const context = new (window.AudioContext || window.webkitAudioContext)(); await context.resume()
    const source = context.createMediaStreamSource(stream)
    const node = context.createScriptProcessor(2048, 1, 1)            // contiguous blocks of samples, no overlap
    const mic = { stream, context, source, node, listener: null }
    node.onaudioprocess = event => { mic.listener?.(event.inputBuffer.getChannelData(0)); event.outputBuffer.getChannelData(0).fill(0) }
    source.connect(node); node.connect(context.destination)
    micRef.current = mic
    return mic
  }, [])

  // Average-and-decimate to 16 kHz: the recogniser wants 16 kHz, and sending it that way is 3x smaller and quicker to read.
  const toSixteenK = useCallback((input, rate) => {
    if (rate <= 16000) return { samples: input, rate }
    const ratio = rate / 16000; const length = Math.floor(input.length / ratio); const out = new Float32Array(length)
    for (let i = 0; i < length; i += 1) {
      const start = Math.floor(i * ratio); const end = Math.min(input.length, Math.max(start + 1, Math.floor((i + 1) * ratio)))
      let sum = 0; for (let j = start; j < end; j += 1) sum += input[j]
      out[i] = sum / (end - start)
    }
    return { samples: out, rate: 16000 }
  }, [])

  const listen = useCallback(async () => {
    setPhase('listening'); setMessage('Listening for your answer'); setLevel(0.18)
    if (demo) {
      await new Promise(resolve => setTimeout(resolve, 1700))
      return { blob: null, heard: DEMO_TURNS[demoIndex.current]?.heard || 'Yes' }
    }
    const mic = await openMic()
    const rate = mic.context.sampleRate
    return new Promise(resolve => {
      const threshold = 0.018; const silenceMs = 800; const maxPre = Math.ceil(rate * 0.3 / 2048)
      const preroll = []; let chunks = []; let started = false; let loudFrames = 0; let lastLoud = 0; const startedAt = performance.now()
      const finish = result => { mic.listener = null; setLevel(0.05); resolve(result) }
      mic.listener = frame => {
        const now = performance.now()
        let sum = 0; for (let i = 0; i < frame.length; i += 1) sum += frame[i] * frame[i]
        const rms = Math.sqrt(sum / frame.length); setLevel(Math.min(1, rms * 8))
        const copy = new Float32Array(frame)
        if (started) chunks.push(copy)
        else { preroll.push(copy); if (preroll.length > maxPre) preroll.shift() }
        if (rms > threshold) {
          loudFrames += 1; lastLoud = now
          if (!started) { started = true; chunks = [...preroll] }
        }
        if (started && now - lastLoud > silenceMs) {
          if (loudFrames < 3) { started = false; chunks = []; loudFrames = 0; return }          // a click or cough, not an answer
          const merged = new Float32Array(chunks.reduce((total, chunk) => total + chunk.length, 0)); let offset = 0
          chunks.forEach(chunk => { merged.set(chunk, offset); offset += chunk.length })
          const { samples, rate: outRate } = toSixteenK(merged, rate)
          return finish({ blob: encodeWav(samples, outRate) })
        }
        if (started && now - startedAt > 25000) return finish({ noSpeech: true })
        if (!started && now - startedAt > 9000) return finish({ noSpeech: true })
      }
    })
  }, [demo, encodeWav, openMic, toSixteenK])

  const runDemo = useCallback(async () => {
    const item = DEMO_TURNS[demoIndex.current]
    if (!item) return
    setTurn(item); addTimeline('svara', item.say); await speak(item.say)
    if (item.done) { setFinished(true); setPhase('done'); setMessage('Form saved successfully'); return }
    setPhase('listening'); setMessage('Listening for your answer'); await new Promise(resolve => setTimeout(resolve, 1200)); setPhase('hearing'); setMessage('I hear you'); await new Promise(resolve => setTimeout(resolve, 500))
    if (item.heard) { setTurn(current => ({ ...current, heard: item.heard })); addTimeline('you', item.heard) }
    const next = DEMO_TURNS[demoIndex.current + 1]
    demoIndex.current += 1
    if (next) { setProgress([Math.min(form.fields.length, Math.ceil((demoIndex.current) / 2)), form.fields.length]); timers.current.push(setTimeout(runDemo, 700)) }
  }, [addTimeline, form.fields.length, speak])

  const runReal = useCallback(async () => {
    try {
      openMic().catch(() => {})                  // ask for the microphone while SVARA says hello
      const first = await jsonApi('/api/call/start', { form_id: form.id, language })
      realCallRef.current = first.call_id
      let current = first.turn; let completed = false
      while (current && !current.done) {
        setTurn(current); addTimeline('svara', current.say); await speak(current.say)
        const heard = await listen()
        setPhase('thinking'); setMessage('Thinking through your answer')
        let response
        if (heard.noSpeech) response = await jsonApi(`/api/call/${first.call_id}/event`, { event: 'no_speech' })
        else { if (heard.blob) { const raw = await heard.blob.arrayBuffer(); response = await api(`/api/call/${first.call_id}/turn`, { method: 'POST', headers: { 'Content-Type': 'audio/wav' }, body: raw }).then(result => result.json()) } else response = { turn: { ...current, heard: heard.heard } } }
        current = response.turn
        if (current.heard) addTimeline('you', current.heard)
        setTurn(current)
        const [done, total] = current.progress || [0, form.fields.length]; setProgress([done, total])
      }
      if (current) { setTurn(current); if (current.say) { addTimeline('svara', current.say); await speak(current.say) } }
      completed = current?.outcome === 'completed' || current?.done
      closeMic(); setFinished(true); setPhase('done'); setMessage(completed ? 'Form saved successfully' : 'Call ended')
    } catch (caught) {
      closeMic(); setError(caught.message || 'The call could not connect.'); setPhase('error'); setMessage('Something needs your attention')
    }
  }, [addTimeline, closeMic, form.fields.length, form.id, language, listen, openMic, speak])

  useEffect(() => {
    if (demo) runDemo()
    else runReal()
    return () => { timers.current.forEach(clearTimeout); window.speechSynthesis?.cancel(); closeMic() }
  }, []) // the call should only start once

  const endCall = async () => {
    timers.current.forEach(clearTimeout); window.speechSynthesis?.cancel(); closeMic()
    if (!demo && realCallRef.current) { try { await jsonApi(`/api/call/${realCallRef.current}/event`, { event: 'hangup' }) } catch { /* the session may already be complete */ } }
    setFinished(true); setPhase('done'); setMessage('Your answers so far are saved.'); setTurn(current => ({ ...current, say: 'The call has ended. Your answers so far are saved.' }))
  }

  return <div className="call-screen"><div className="call-screen__header"><div className="brand brand--light"><BrandMark /><div><strong>SVARA</strong><span>voice-first forms</span></div></div><div className="call-meta"><span className={`live-badge ${demo ? 'live-badge--demo' : ''}`}><i />{demo ? 'Demo mode' : 'Live call'}</span><span>{form.title}</span><span>{LANG_NAMES[language]}</span></div><button className="call-close" onClick={onClose} aria-label="Close call"><Icon name="close" size={20} /></button></div><main className="call-main"><div className="call-main__left"><div className="call-eyebrow"><span className="pulse-dot" />HANDS-FREE CONVERSATION</div><SignalScene size={330} amp={phase === 'speaking' || phase === 'hearing' ? 1 : phase === 'listening' ? 0.3 + level : 0.2}><CallOrb phase={phase} /></SignalScene><div className="call-state"><span className={`state-tag state-tag--${phase}`}>{phase === 'speaking' ? 'Speaking' : phase === 'listening' ? 'Listening' : phase === 'hearing' ? 'Hearing you' : phase === 'thinking' ? 'Thinking' : phase === 'done' ? 'Complete' : phase === 'error' ? 'Needs attention' : 'Ready'}</span><h2>{message}</h2><p>{phase === 'speaking' ? 'Listen for the next question.' : phase === 'listening' || phase === 'hearing' ? 'Take your time. There is no need to press anything.' : phase === 'thinking' ? 'Turning your words into a clear answer.' : phase === 'done' ? 'Your conversation has been stored safely.' : error || 'We are checking the connection.'}</p></div><div className="voice-meter"><span>MICROPHONE LEVEL</span><div>{Array.from({ length: 28 }, (_, index) => <i key={index} style={{ height: `${Math.max(3, (level * (18 + Math.sin(index * 1.7) * 14)))}px` }} />)}</div></div>{!finished && <button className="end-call" onClick={endCall}>End call</button>}{finished && form.has_pdf && realCallRef.current && <button className="button button--outline-light" onClick={() => { setPdfNote(''); downloadFilled(realCallRef.current, form.title).catch(caught => setPdfNote(caught.message || 'The filled form could not be created.')) }}><Icon name="download" size={16} />Download filled form</button>}{pdfNote && <p className="call-pdf-note" role="alert">{pdfNote}</p>}{finished && <button className="button button--bone" onClick={onClose}>Back to workspace <Icon name="arrow" size={16} /></button>}</div><div className="call-main__right"><div className="call-panel__header"><div><span className="section-kicker">LIVE TRANSCRIPT</span><h3>Your conversation</h3></div><span className="progress-copy">{progress[0]} / {progress[1]}</span></div><div className="call-progress"><span style={{ width: `${progress[1] ? Math.min(100, progress[0] / progress[1] * 100) : 0}%` }} /></div><div className="timeline" aria-live="polite">{timeline.map(item => <div className={`timeline-item timeline-item--${item.who}`} key={item.id}><span className="timeline-avatar">{item.who === 'svara' ? <BrandMark /> : userInitials}</span><div><span className="timeline-who">{item.who === 'svara' ? 'SVARA' : 'YOU'}</span><p>{item.text}</p></div></div>)}{!timeline.length && <div className="timeline-empty"><Icon name="wave" size={20} /><p>Your transcript will appear here as the call moves forward.</p></div>}</div><div className="privacy-note"><Icon name="shield" size={16} /><span>Stored locally after every turn · audio is never kept by default</span></div></div></main></div>
}

function CallDetail({ call, onClose, connected }) {
  const [record, setRecord] = useState(null)
  const [note, setNote] = useState('')
  const real = connected && Boolean(call.call_id)
  useEffect(() => {
    if (!real) return
    api(`/api/call/${call.call_id}`).then(response => response.json()).then(setRecord).catch(() => setNote('Could not load the answers for this call.'))
  }, [call.call_id, real])
  const answers = (record?.fields || []).filter(field => field.value !== undefined && field.value !== null && field.value !== '')
  const pretty = name => name.replace(/_/g, ' ').replace(/^./, letter => letter.toUpperCase())
  const downloadPdf = async () => {
    setNote('')
    try {
      const blob = await (await api(`/api/call/${call.call_id}/filled.pdf`)).blob()
      const link = document.createElement('a'); link.href = URL.createObjectURL(blob); link.download = `${call.form_title || 'form'}-filled.pdf`; link.click(); URL.revokeObjectURL(link.href)
    } catch (caught) { setNote(caught.message || 'The filled form could not be created.') }
  }
  return <div className="drawer-backdrop" onClick={onClose}><aside className="detail-drawer" onClick={event => event.stopPropagation()}><div className="drawer-head"><div><span className="section-kicker">RESPONSE DETAIL</span><h2>{call.form_title}</h2><span className="drawer-id">Call #{call.id}</span></div><button className="modal-close" onClick={onClose} aria-label="Close details"><Icon name="close" size={18} /></button></div><div className={`detail-outcome detail-outcome--${call.outcome}`}><span className="detail-outcome__icon"><Icon name={call.outcome === 'completed' ? 'check' : 'pulse'} size={18} /></span><div><strong>{statusLabel(call.outcome)}</strong><span>{call.outcome === 'completed' ? 'The form was confirmed and saved.' : 'This conversation may need a follow-up.'}</span></div></div><div className="detail-grid"><div><span>Started</span><strong>{formatDate(call.started_at)}</strong></div><div><span>Duration</span><strong>{call.duration || '—'}</strong></div><div><span>Language</span><strong>{LANG_NAMES[call.language] || call.language}</strong></div><div><span>Fields completed</span><strong>{call.fields_completed ?? '—'}</strong></div></div><div className="detail-section"><span className="section-kicker">CAPTURED ANSWERS</span><div className="answer-list">{real ? (record ? (answers.length ? answers.map(field => <div className="answer-row" key={field.name}><span>{field.label || pretty(field.name)}</span><strong>{String(field.value)}</strong><Icon name={field.status === 'confirmed' ? 'check' : 'clock'} size={15} /></div>) : <div className="answer-row"><span>No answers were captured in this call.</span></div>) : <div className="answer-row"><span>{note || 'Loading answers…'}</span></div>) : ['Full name', 'Age', 'Mobile number', 'Blood group'].map((label, index) => <div className="answer-row" key={label}><span>{label}</span><strong>{index === 0 ? 'Anil Kumar' : index === 1 ? '25' : index === 2 ? '•••• 3210' : 'O+'}</strong><Icon name="check" size={15} /></div>)}</div></div>{real && <button className="button button--outline button--wide" onClick={downloadPdf}><Icon name="download" size={16} />Download filled PDF</button>}{real && note && record && <p className="detail-note" role="alert">{note}</p>}</aside></div>
}

function UploadFormModal({ connected, onClose, onPublished, personal = false }) {
  const [stage, setStage] = useState('pick')
  const [error, setError] = useState('')
  const [drag, setDrag] = useState(false)
  const [result, setResult] = useState(null)
  const [title, setTitle] = useState('')
  const [fields, setFields] = useState([])
  const [busy, setBusy] = useState(false)
  const [fileName, setFileName] = useState('')
  const [published, setPublished] = useState(null)
  const input = useRef(null)

  const readFile = async file => {
    if (!file) return
    if (!connected) { setError('The SVARA server is not connected. Start it with: python -m callpage'); return }
    if (!/\.pdf$/i.test(file.name) && file.type !== 'application/pdf') { setError('Please choose a PDF file.'); return }
    if (file.size > 12 * 1024 * 1024) { setError('That PDF is larger than 12 MB.'); return }
    setError(''); setFileName(file.name); setStage('reading')
    try {
      const response = await api('/api/admin/forms/analyze', { method: 'POST', headers: { 'Content-Type': 'application/pdf', 'X-Filename': file.name.replace(/[^\x20-\x7e]/g, '_') }, body: file })
      const data = await response.json()
      setResult(data); setTitle(data.title); setFields(data.fields); setStage('review')
    } catch (caught) { setError(caught.message || 'The form could not be read.'); setStage('pick') }
  }

  const update = (id, key, value) => setFields(current => current.map(field => field.id === id ? { ...field, [key]: value } : field))
  const asked = fields.filter(field => !field.ignored).length
  const publish = async () => {
    setBusy(true); setError('')
    try {
      const saved = await jsonApi('/api/admin/forms/publish', { template_id: result.template_id, title, fields: fields.map(field => ({ id: field.id, label: field.label, call_name: field.call_name, call_kind: field.call_kind, required: field.required, ignored: field.ignored, question: field.question })) })
      setPublished(saved); onPublished(saved); setStage('done')
    } catch (caught) { setError(caught.message || 'The form could not be published.') } finally { setBusy(false) }
  }

  return <div className="modal-backdrop"><section className="editor-modal upload-modal" role="dialog" aria-modal="true" aria-labelledby="upload-title">
    <div className="editor-header"><div><span className="section-kicker">UPLOAD A FORM</span><h2 id="upload-title">{stage === 'done' ? (personal ? 'Your form is ready.' : 'Your form is live.') : stage === 'review' ? 'Check what SVARA will ask.' : personal ? 'Fill in your own form by voice.' : 'Turn a PDF into a conversation.'}</h2><p>{stage === 'review' ? `${asked} of ${fields.length} blanks will be asked. Edit any question, or switch a blank off.` : stage === 'done' ? (personal ? 'Find it under your forms, answer by voice, then download it filled in.' : 'Callers can now choose it and answer by voice.') : personal ? 'Upload the form you need to complete. Only you can see it.' : 'SVARA reads the blanks on a digital PDF and writes a spoken question for each one.'}</p></div><button className="modal-close" onClick={onClose} aria-label="Close upload"><Icon name="close" size={18} /></button></div>
    {stage === 'pick' && <div className="upload-body"><div className={`upload-drop ${drag ? 'is-drag' : ''}`} onDragOver={event => { event.preventDefault(); setDrag(true) }} onDragLeave={() => setDrag(false)} onDrop={event => { event.preventDefault(); setDrag(false); readFile(event.dataTransfer.files?.[0]) }}><span className="upload-drop__icon"><Icon name="download" size={26} /></span><strong>Drop your form here</strong><span>or</span><button className="button button--primary" onClick={() => input.current?.click()}>Choose a PDF</button><input ref={input} type="file" accept="application/pdf,.pdf" hidden onChange={event => { readFile(event.target.files?.[0]); event.target.value = '' }} /><small>Digital PDFs only (not scans) · up to 12 MB · read on this computer</small></div>{error && <p className="upload-error" role="alert">{error}</p>}</div>}
    {stage === 'reading' && <div className="upload-body upload-reading" aria-live="polite"><div className="upload-scan"><i /><i /><i /><i /><i /></div><strong>Reading {fileName}…</strong><span>Finding the blanks, their labels and what each one means.</span></div>}
    {stage === 'review' && <><div className="upload-body upload-review"><div className="upload-review__top"><div><label className="field-label" htmlFor="upload-title-input">Form name</label><input id="upload-title-input" className="editor-input editor-input--large" value={title} onChange={event => setTitle(event.target.value)} /></div><div className="upload-stats"><span><b>{result.pages}</b>page{result.pages === 1 ? '' : 's'}</span><span><b>{fields.length}</b>blanks found</span><span><b>{asked}</b>to ask</span></div></div>{result.warnings?.length > 0 && <ul className="upload-warnings">{result.warnings.map(warning => <li key={warning}>{warning}</li>)}</ul>}<div className="review-list">{fields.map(field => <div className={`review-row ${field.ignored ? 'is-off' : ''}`} key={field.id}><label className="review-ask"><input type="checkbox" checked={!field.ignored} onChange={event => update(field.id, 'ignored', !event.target.checked)} /><span>Ask</span></label><div className="review-main"><div className="review-line"><input className="editor-input" value={field.label} onChange={event => update(field.id, 'label', event.target.value)} aria-label="Field label" /><select className="editor-input editor-input--kind" value={field.call_kind} onChange={event => update(field.id, 'call_kind', event.target.value)} aria-label="Answer type">{[...new Set([field.call_kind, ...(result.kinds || [])])].map(kind => <option key={kind} value={kind}>{kind.replace(/_/g, ' ')}</option>)}</select></div>{!field.ignored && <input className="editor-input editor-input--question" value={field.question} onChange={event => update(field.id, 'question', event.target.value)} aria-label="Spoken question" placeholder="SVARA will word this question itself" />}<div className="review-meta"><span>Page {field.page} · {field.shape}</span><label className="required-toggle"><input type="checkbox" checked={field.required} onChange={event => update(field.id, 'required', event.target.checked)} disabled={field.ignored} /> Required</label><span className={`confidence confidence--${field.confidence >= 0.75 ? 'high' : field.confidence >= 0.5 ? 'mid' : 'low'}`}>{Math.round(field.confidence * 100)}% sure</span></div></div></div>)}</div></div><div className="editor-footer"><span>{error ? <b className="editor-error" role="alert">{error}</b> : 'Nothing leaves this computer.'}</span><div><button className="button button--outline" onClick={() => { setStage('pick'); setResult(null) }}>Choose another file</button><button className="button button--primary" disabled={busy || asked === 0} onClick={publish}>{busy ? 'Saving…' : personal ? 'Save my form' : 'Publish form'} {!busy && <Icon name="arrow" size={16} />}</button></div></div></>}
    {stage === 'done' && <div className="upload-body upload-reading"><span className="upload-done"><Icon name="check" size={30} /></span><strong>{title}</strong><span>{asked} questions are ready{personal ? '.' : ' for callers.'}</span>{published?.access_code && <div className="code-reveal"><span>Access code for callers</span><strong>{published.access_code}</strong><small>Share it with the people who should fill this form. Change it any time from the form's ••• menu.</small></div>}<button className="button button--primary" onClick={onClose}>Done <Icon name="arrow" size={16} /></button></div>}
  </section></div>
}

function ConfirmDialog({ title, body, confirmLabel, busy, error, onConfirm, onCancel }) {
  return <div className="modal-backdrop" onClick={onCancel}><section className="confirm-dialog" role="alertdialog" aria-modal="true" aria-labelledby="confirm-title" onClick={event => event.stopPropagation()}>
    <span className="confirm-dialog__icon"><Icon name="close" size={22} /></span>
    <h2 id="confirm-title">{title}</h2><p>{body}</p>{error && <p className="upload-error" role="alert">{error}</p>}
    <div className="confirm-dialog__actions"><button className="button button--outline" onClick={onCancel} disabled={busy} autoFocus>Keep it</button><button className="button button--danger" onClick={onConfirm} disabled={busy}>{busy ? 'Deleting…' : confirmLabel}</button></div>
  </section></div>
}

function Switch({ checked, onChange, label, hint }) {
  return <label className="switch-row"><span><strong>{label}</strong>{hint && <small>{hint}</small>}</span><input type="checkbox" role="switch" checked={checked} onChange={event => onChange(event.target.checked)} /><i className="switch" aria-hidden="true" /></label>
}

function SettingsView({ session, forms, connected, onRename, motionOff, setMotionOff, onSettings, onLogout }) {
  const [status, setStatus] = useState(null)
  const [name, setName] = useState(session.name)
  const [renamed, setRenamed] = useState(false)
  const [formId, setFormId] = useState(forms[0]?.id || '')
  const [message, setMessage] = useState('')
  const form = forms.find(item => item.id === formId) || forms[0]
  const settings = form?.settings || {}
  useEffect(() => {
    if (!connected) return
    api('/api/admin/settings').then(response => response.json()).then(setStatus).catch(() => setStatus({ error: true }))
  }, [connected])
  const change = async patch => {
    setMessage('')
    if (!connected) { setMessage('Connect the SVARA server to change privacy settings.'); return }
    try { const response = await jsonApi('/api/admin/forms/settings', { id: form.id, settings: patch }); onSettings(form.id, response.settings); setMessage('Saved') } catch (caught) { setMessage(caught.message || 'Could not save.') }
  }
  const saveName = event => { event.preventDefault(); if (name.trim().length < 2) return; onRename(name.trim().slice(0, 40)); setRenamed(true); setTimeout(() => setRenamed(false), 1800) }
  const speech = status?.speech || {}
  return <div className="view-stack settings">
    <section className="settings-card">
      <div className="settings-card__head"><span className="section-kicker">ACCOUNT</span><h3>Your profile</h3></div>
      <form className="settings-form" onSubmit={saveName}><label className="field-label" htmlFor="settings-name">Display name</label><div className="settings-inline"><input id="settings-name" className="editor-input editor-input--large" value={name} onChange={event => setName(event.target.value)} /><button className="button button--primary" type="submit">{renamed ? 'Saved' : 'Save name'}</button></div></form>
      <div className="settings-meta"><span>Role</span><strong>Administrator</strong></div>
      <div className="settings-meta"><span>Server</span><strong className={connected ? 'is-ok' : 'is-bad'}>{connected ? 'Connected' : 'Not connected (demo workspace)'}</strong></div>
      <button className="button button--outline" onClick={onLogout}>Sign out</button>
    </section>

    <section className="settings-card">
      <div className="settings-card__head"><span className="section-kicker">PRIVACY</span><h3>How a form handles calls</h3></div>
      {form ? <>
        <label className="field-label" htmlFor="settings-form">Form</label>
        <div className="select-wrap"><select id="settings-form" value={form.id} onChange={event => { setFormId(event.target.value); setMessage('') }}>{forms.map(item => <option key={item.id} value={item.id}>{item.title}</option>)}</select><Icon name="chevron" size={16} /></div>
        <Switch checked={settings.require_consent !== false} onChange={value => change({ require_consent: value })} label="Ask for consent first" hint="SVARA asks the caller to agree before any question." />
        <Switch checked={Boolean(settings.store_audio)} onChange={value => change({ store_audio: value })} label="Keep audio recordings" hint="Only saved when the caller also agrees. Off by default." />
        <Switch checked={settings.redact_sensitive_log !== false} onChange={value => change({ redact_sensitive_log: value })} label="Hide sensitive answers in logs" hint="Aadhaar, phone and similar answers are masked." />
        <div className="switch-row"><span><strong>Re-ask an unclear answer</strong><small>How many times before SVARA moves on.</small></span><select className="editor-input editor-input--kind" value={settings.max_attempts || 3} onChange={event => change({ max_attempts: Number(event.target.value) })} aria-label="Attempts">{[1, 2, 3, 4, 5, 6].map(n => <option key={n} value={n}>{n} time{n === 1 ? '' : 's'}</option>)}</select></div>
        <p className={`settings-message ${message && message !== 'Saved' ? 'is-error' : ''}`} role="status">{message}</p>
      </> : <p className="settings-empty">No forms yet.</p>}
    </section>

    <section className="settings-card">
      <div className="settings-card__head"><span className="section-kicker">SPEECH ENGINE</span><h3>How SVARA listens and speaks</h3></div>
      {!connected ? <p className="settings-empty">Start the server with <code>python -m callpage</code> to see its status.</p> : !status ? <p className="settings-empty">Checking…</p> : <>
        <div className="settings-meta"><span>Recognition</span><strong>{speech.cloud_ready ? `Cloud (${speech.provider})` : `This computer · Whisper ${speech.local_model || ''}`}</strong></div>
        {!speech.cloud_ready && <p className="settings-tip">Answers are quicker with a cloud key: add <code>GROQ_API_KEY=…</code> to a <code>.env</code> file and restart. Or run with <code>--fast</code>.</p>}
        <div className="settings-meta"><span>Decoding</span><strong>{speech.beam === 1 ? 'Fast (greedy)' : `Careful (beam ${speech.beam})`} · {speech.threads || 'auto'} threads</strong></div>
        <div className="settings-meta"><span>Server voice</span><strong>{status.server_voice ? 'On' : 'Off · using the browser voice'}</strong></div>
        <div className="settings-meta"><span>Forms · calls stored</span><strong>{status.forms} · {status.calls}</strong></div>
        <div className="settings-meta settings-meta--path"><span>Saved in</span><strong>{status.calls_dir}</strong></div>
      </>}
    </section>

    <section className="settings-card">
      <div className="settings-card__head"><span className="section-kicker">APPEARANCE</span><h3>Motion</h3></div>
      <Switch checked={motionOff} onChange={setMotionOff} label="Reduce motion" hint="Turns off animated transitions, the spinning 3D ring and scroll effects." />
    </section>
  </div>
}

function EmptyForms() {
  return <div className="empty-forms"><span className="empty-forms__icon"><Icon name="lock" size={22} /></span><strong>No forms yet</strong><p>Forms appear here once you enter the access code you were given. Type it in the box above.</p></div>
}

function AccessCodeCard({ onUnlock, onStart }) {
  const [code, setCode] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [opened, setOpened] = useState(null)
  const submit = async event => {
    event.preventDefault()
    if (code.replace(/[^A-Za-z0-9]/g, '').length < 4) { setError('Enter the access code you were given.'); return }
    setBusy(true); setError('')
    try { setOpened(await onUnlock(code)); setCode('') } catch (caught) { setError(caught.message || 'Could not check the code.') } finally { setBusy(false) }
  }
  return <section className="code-card" aria-labelledby="code-card-title">
    <span className="code-card__icon"><Icon name="lock" size={24} /></span>
    <div className="code-card__text"><span className="section-kicker">ACCESS CODE</span><h3 id="code-card-title">Were you given a code?</h3><p>Type it here to open the form it belongs to. Only forms you have a code for appear in your list.</p></div>
    {opened ? <div className="code-card__done" role="status"><strong>{opened.length === 1 ? 'Form unlocked' : `${opened.length} forms unlocked`}</strong>{opened.map(form => <div className="code-card__form" key={form.id}><span>{form.title}</span><button className="button button--primary button--small" onClick={() => onStart(form)}>Fill it now <Icon name="arrow" size={15} /></button></div>)}<button className="text-button" onClick={() => setOpened(null)}>Enter another code</button></div>
      : <form className="code-card__form-row" onSubmit={submit} noValidate><input className="editor-input editor-input--large code-input code-input--card" value={code} onChange={event => { setCode(event.target.value.toUpperCase().replace(/[^A-Z0-9-]/g, '').slice(0, 17)); setError('') }} placeholder="ACCESS CODE" autoComplete="off" autoCapitalize="characters" spellCheck="false" aria-label="Access code" /><button className="button button--primary" type="submit" disabled={busy}>{busy ? 'Checking…' : 'Open form'} {!busy && <Icon name="arrow" size={16} />}</button><div className="upload-error code-card__error" role="alert">{error}</div></form>}
  </section>
}

function AccessDialog({ form, onClose, onChanged }) {
  const [code, setCode] = useState(form.access_code || '')
  const [custom, setCustom] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [copied, setCopied] = useState(false)
  const act = async (action, extra = {}) => {
    setBusy(true); setError(''); setCopied(false)
    try { const result = await jsonApi('/api/admin/forms/access', { id: form.id, action, ...extra }); setCode(result.access_code || ''); setCustom(''); onChanged(form.id, result.access_code || null) } catch (caught) { setError(caught.message || 'Could not change the code.') } finally { setBusy(false) }
  }
  const copy = async () => { try { await navigator.clipboard.writeText(code); setCopied(true); setTimeout(() => setCopied(false), 1500) } catch { setError('Copy is not available here. Select the code and copy it.') } }
  return <div className="modal-backdrop" onClick={onClose}><section className="confirm-dialog code-dialog" role="dialog" aria-modal="true" aria-labelledby="access-title" onClick={event => event.stopPropagation()}>
    <span className="confirm-dialog__icon confirm-dialog__icon--lock"><Icon name="lock" size={22} /></span>
    <h2 id="access-title">Access code</h2><p>{form.title}</p>
    {code ? <div className="code-reveal"><span>Callers must enter</span><strong>{code}</strong><button className="text-button" onClick={copy}>{copied ? 'Copied' : 'Copy code'}</button></div> : <div className="code-reveal code-reveal--open"><span>No code is set</span><small>Any signed-in caller can open this form.</small></div>}
    <p className="access-note">Changing the code asks everyone to enter the new one again. Callers never see the code on screen: you share it with them yourself.</p>
    <div className="access-actions"><button className="button button--outline button--small" onClick={() => act('generate')} disabled={busy}>{code ? 'New random code' : 'Create a code'}</button>{code && <button className="button button--outline button--small" onClick={() => act('clear')} disabled={busy}>Remove code</button>}</div>
    <div className="access-custom"><input className="editor-input" value={custom} onChange={event => { setCustom(event.target.value.toUpperCase().replace(/[^A-Z0-9-]/g, '').slice(0, 17)); setError('') }} placeholder="Or type your own (4–16 letters or digits)" aria-label="Custom access code" /><button className="button button--primary button--small" onClick={() => act('set', { code: custom })} disabled={busy || custom.replace(/[^A-Z0-9]/g, '').length < 4}>Use it</button></div>
    <p className="upload-error" role="alert">{error}</p>
    <div className="confirm-dialog__actions"><button className="button button--outline" onClick={onClose}>Close</button></div>
  </section></div>
}

function FormEditor({ form, onClose, onSave }) {
  const editing = Boolean(form)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [title, setTitle] = useState(form?.title || '')
  const [description, setDescription] = useState(form?.description || '')
  const [eyebrow, setEyebrow] = useState(form?.eyebrow || 'Custom flow')
  const [accent, setAccent] = useState(form?.accent || 'green')
  const [fields, setFields] = useState((form?.fields || []).map(field => ({ kind: 'text', question: '', ...field })))

  const updateField = (index, key, value) => setFields(current => current.map((field, itemIndex) => itemIndex === index ? { ...field, [key]: value } : field))
  const moveField = (index, direction) => setFields(current => { const next = [...current]; const target = index + direction; if (target < 0 || target >= next.length) return current; [next[index], next[target]] = [next[target], next[index]]; return next })
  const addField = () => setFields(current => [...current, { name: `custom_${Date.now()}`, label: 'New question', question: 'What would you like to share?', kind: 'text', required: false }])
  const save = async published => {
    if (!title.trim()) { setError('Give the form a name.'); return }
    if (!fields.length) { setError('Add at least one question.'); return }
    setBusy(true); setError('')
    try { await onSave({ id: form?.id || `custom_${Date.now()}`, title: title.trim(), description: description.trim() || 'A thoughtful voice-first flow for your callers.', eyebrow: eyebrow.trim() || 'Custom flow', accent, language: 'en', languages: form?.languages || ['en'], fields }, published) } catch (caught) { setError(caught.message || 'Could not save the form.') } finally { setBusy(false) }
  }

  return <div className="modal-backdrop"><section className="editor-modal" role="dialog" aria-modal="true" aria-labelledby="editor-title"><div className="editor-header"><div><span className="section-kicker">{editing ? 'EDIT FORM' : 'NEW FORM'}</span><h2 id="editor-title">{editing ? 'Tune the conversation.' : 'Design a new flow.'}</h2><p>Every field becomes a spoken question. Keep it short, clear, and human.</p></div><button className="modal-close" onClick={onClose} aria-label="Close editor"><Icon name="close" size={18} /></button></div><div className="editor-body"><div className="editor-details"><label className="field-label" htmlFor="form-title">Form name</label><input id="form-title" className="editor-input editor-input--large" value={title} onChange={event => setTitle(event.target.value)} placeholder="e.g. New patient intake" /><label className="field-label" htmlFor="form-eyebrow">Category label</label><input id="form-eyebrow" className="editor-input" value={eyebrow} onChange={event => setEyebrow(event.target.value)} placeholder="Healthcare" /><label className="field-label" htmlFor="form-description">Description</label><textarea id="form-description" className="editor-input editor-textarea" value={description} onChange={event => setDescription(event.target.value)} placeholder="What is this flow for?" /><label className="field-label" htmlFor="form-accent">Visual accent</label><div className="accent-options"><select id="form-accent" className="editor-input" value={accent} onChange={event => setAccent(event.target.value)}><option value="green">Ink + aqua</option><option value="burgundy">Coral + burgundy</option><option value="bone">Bone + mint</option></select><span className={`accent-preview accent-preview--${accent}`} /></div><div className="editor-tip"><Icon name="spark" size={16} /><span>Published forms appear instantly in the caller experience.</span></div></div><div className="editor-fields"><div className="editor-fields__head"><div><span className="section-kicker">QUESTION ORDER</span><h3>{fields.length} fields in this flow</h3></div><button className="button button--outline button--small" onClick={addField}><Icon name="plus" size={15} />Add field</button></div><div className="field-list">{fields.map((field, index) => <div className="field-editor-row" key={`${field.name}-${index}`}><div className="field-order"><button onClick={() => moveField(index, -1)} disabled={index === 0} aria-label="Move field up">↑</button><span>{String(index + 1).padStart(2, '0')}</span><button onClick={() => moveField(index, 1)} disabled={index === fields.length - 1} aria-label="Move field down">↓</button></div><div className="field-editor-main"><input className="editor-input" value={field.label} onChange={event => updateField(index, 'label', event.target.value)} aria-label="Field label" /><input className="editor-input editor-input--question" value={field.question || ''} onChange={event => updateField(index, 'question', event.target.value)} aria-label="Spoken question" placeholder="Spoken question" /><div className="field-editor-meta"><select className="editor-input editor-input--kind" value={field.kind || 'text'} onChange={event => updateField(index, 'kind', event.target.value)} aria-label="Field type"><option value="text">Short answer</option><option value="name">Name</option><option value="number">Number</option><option value="age">Age</option><option value="phone">Phone number</option><option value="email">Email</option><option value="date">Date</option><option value="date_of_birth">Date of birth</option><option value="yes_no">Yes / no</option><option value="gender">Gender</option><option value="address">Address</option><option value="pincode">PIN code</option><option value="aadhaar">Aadhaar number</option><option value="blood_group">Blood group</option></select><label className="required-toggle"><input type="checkbox" checked={Boolean(field.required)} onChange={event => updateField(index, 'required', event.target.checked)} /> Required</label><button className="field-remove" onClick={() => setFields(current => current.filter((_, itemIndex) => itemIndex !== index))}>Remove</button></div></div></div>)}</div></div></div><div className="editor-footer"><span>{error ? <b className="editor-error" role="alert">{error}</b> : <><span className="pulse-dot" /> Publishing makes the form available to callers</>}</span><div><button className="button button--outline" disabled={busy} onClick={() => save(false)}>Save draft</button><button className="button button--primary" disabled={busy} onClick={() => save(true)}>{busy ? 'Saving…' : editing ? 'Publish changes' : 'Publish form'} {!busy && <Icon name="arrow" size={16} />}</button></div></div></section></div>
}


// ---------------------------------------------------------------------------------------------------------------
// Sign-in. NOTE: this gates the interface only; the Python call server has no user accounts of its own.
const SESSION_KEY = 'svara.session'
function loadSession() {
  try {
    const value = JSON.parse(window.sessionStorage.getItem(SESSION_KEY))
    if (value && (value.role === 'admin' || value.role === 'user') && value.name && value.token) { authToken = value.token; return value }
  } catch { /* nothing stored */ }
  return null
}
function saveSession(value) {
  authToken = value?.token || ''
  try { if (value) window.sessionStorage.setItem(SESSION_KEY, JSON.stringify(value)); else window.sessionStorage.removeItem(SESSION_KEY) } catch { /* storage unavailable: the session lasts until reload */ }
}

function AuthBrand({ light }) { return <div className={`auth-brand ${light ? 'auth-brand--light' : ''}`}><BrandMark /><strong>SVARA</strong></div> }

function RoleChooser({ go }) {
  const tiltAdmin = useTilt(12)
  const tiltCaller = useTilt(12)
  return <div className="auth auth--chooser page-transition"><div className="auth-orbs"><i /><i /><i /></div>
    <AuthBrand />
    <main className="chooser">
      <div className="chooser-copy"><div className="eyebrow eyebrow--dark"><span className="eyebrow-line" />WELCOME</div><h1>Who is joining SVARA today?</h1><p>Two doors, one voice. Choose how you want to use SVARA and sign in to your own space.</p></div>
      <div className="chooser-scene"><SignalScene size={300}><Orb compact /></SignalScene></div>
      <div className="chooser-cards">
        <button className="role-card role-card--admin" {...tiltAdmin} onClick={() => go('/login/admin')}><span className="role-card__icon"><Icon name="settings" size={22} /></span><strong>I am an admin</strong><p>Design forms, tune every spoken question, publish, and review every response.</p><span className="role-card__cta">Admin sign in <Icon name="arrow" size={16} /></span></button>
        <button className="role-card role-card--caller" {...tiltCaller} onClick={() => go('/login/caller')}><span className="role-card__icon"><Icon name="mic" size={22} /></span><strong>I need to fill a form</strong><p>Choose a form, pick your language and answer by talking. No typing needed.</p><span className="role-card__cta">Continue as caller <Icon name="arrow" size={16} /></span></button>
      </div>
    </main></div>
}

// Sign in / create an account (name + user name + password, no e-mail, no code), shared by the admin and caller pages.
function useAccountFlow(role, onLogin, startMode = 'signin') {
  const [mode, setMode] = useState(startMode)                      // signin | signup
  const [name, setName] = useState('')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [agree, setAgree] = useState(false)
  const [show, setShow] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [busy, setBusy] = useState(false)
  const [shake, setShake] = useState(0)
  const clear = () => { setError(''); setNotice('') }
  const run = async work => { setBusy(true); clear(); try { await work() } catch (caught) { setError(caught.message || 'Something went wrong.'); setShake(count => count + 1) } finally { setBusy(false) } }

  const signIn = event => { event.preventDefault(); run(async () => {
    if (!username.trim() || !password) throw new Error('Enter your user name and password.')
    const data = await authCall('login', { username, password, role })
    onLogin({ role: data.user.role, name: data.user.name, username: data.user.username, token: data.token })
  }) }
  const signUp = event => { event.preventDefault(); run(async () => {
    if (name.trim().length < 2) throw new Error('Please tell us your name.')
    if (!/^[A-Za-z0-9][A-Za-z0-9_.-]{2,29}$/.test(username.trim())) throw new Error('A user name needs 3 to 30 letters, digits, dots, dashes or underscores, with no spaces or @. For example: athira.admin')
    if (password.length < 6) throw new Error('Choose a password of at least 6 characters.')
    if (password !== confirm) throw new Error('The two passwords do not match.')
    if (role === 'user' && !agree) throw new Error('Please agree to let SVARA use your microphone during calls.')
    await authCall('signup', { name, username, password, role })
    setMode('signin'); setPassword(''); setConfirm(''); setNotice(`Your account is ready. Sign in as "${username.trim().toLowerCase()}" with your password.`)
  }) }
  const pick = next => { setMode(next); clear() }
  return { mode, pick, setMode, name, setName, username, setUsername, password, setPassword, confirm, setConfirm, agree, setAgree, show, setShow, error, notice, busy, shake, clear, submit: mode === 'signin' ? signIn : signUp }
}

function AccountForm({ flow, role, allowSignup = true, setupNote, go }) {
  const f = flow
  const admin = role === 'admin'
  const heading = f.mode === 'signup' ? (admin ? ['FIRST-TIME SETUP', 'Create the administrator account.'] : ['NEW HERE', 'Make your account.'])
    : (admin ? ['ADMINISTRATOR', 'Admin sign in'] : ['WELCOME BACK', 'Let us get you talking.'])
  const lead = f.mode === 'signup' ? (admin ? 'This is the only time an administrator account can be created. Choose a user name and a password.' : 'Choose a user name and a password. That is all you need.')
    : (admin ? 'Only the administrator can manage forms and see responses.' : 'Sign in with your user name and password.')
  return <form key={f.shake} className={`auth-card ${admin ? '' : 'auth-card--caller'} ${f.shake ? 'is-shaking' : ''}`} onSubmit={f.submit} noValidate>
    <button type="button" className="text-button auth-back" onClick={() => go('/login')}>← All sign-ins</button>
    {allowSignup && <div className="auth-tabs" role="tablist"><button type="button" role="tab" aria-selected={f.mode === 'signin'} className={f.mode === 'signin' ? 'is-active' : ''} onClick={() => f.pick('signin')}>Sign in</button><button type="button" role="tab" aria-selected={f.mode === 'signup'} className={f.mode === 'signup' ? 'is-active' : ''} onClick={() => f.pick('signup')}>{admin ? 'Set up admin' : 'Create account'}</button></div>}
    <span className="section-kicker">{heading[0]}</span><h1>{heading[1]}</h1><p className="auth-lead">{lead}</p>
    {setupNote && f.mode === 'signin' && <p className="auth-lead auth-lead--note">{setupNote}</p>}
    {f.mode === 'signup' && <><label className="field-label" htmlFor="acct-name">Your name</label><input id="acct-name" className="editor-input editor-input--large" value={f.name} onChange={event => { f.setName(event.target.value); f.clear() }} autoComplete="name" placeholder="e.g. Anil Kumar" autoFocus /></>}
    <label className="field-label" htmlFor="acct-username">User name</label><input id="acct-username" className="editor-input editor-input--large" value={f.username} onChange={event => { f.setUsername(event.target.value); f.clear() }} autoComplete="username" autoCapitalize="none" spellCheck="false" placeholder={f.mode === 'signup' ? (admin ? 'e.g. athira.admin' : 'e.g. anil.kumar') : 'your user name'} autoFocus={f.mode === 'signin' && !f.notice} />
    {f.mode === 'signup' && <small className="field-hint">Letters, digits, dots, dashes or underscores. No spaces.</small>}
    <label className="field-label" htmlFor="acct-password">Password</label><div className="password-wrap"><input id="acct-password" className="editor-input editor-input--large" type={f.show ? 'text' : 'password'} value={f.password} onChange={event => { f.setPassword(event.target.value); f.clear() }} autoComplete={f.mode === 'signin' ? 'current-password' : 'new-password'} placeholder={f.mode === 'signup' ? 'At least 6 characters' : ''} autoFocus={f.mode === 'signin' && Boolean(f.notice)} /><button type="button" className="text-button" onClick={() => f.setShow(value => !value)}>{f.show ? 'Hide' : 'Show'}</button></div>
    {f.mode === 'signup' && <><label className="field-label" htmlFor="acct-confirm">Confirm password</label><input id="acct-confirm" className="editor-input editor-input--large" type={f.show ? 'text' : 'password'} value={f.confirm} onChange={event => { f.setConfirm(event.target.value); f.clear() }} autoComplete="new-password" />
      {!admin && <label className="consent"><input type="checkbox" checked={f.agree} onChange={event => { f.setAgree(event.target.checked); f.clear() }} /><span>I agree to let SVARA use my microphone during calls. Audio is not kept by default.</span></label>}</>}
    {f.notice && <div className="auth-notice" role="status">{f.notice}</div>}
    <div className="auth-error" role="alert">{f.error}</div>
    {f.mode === 'signin' && f.error && /not right/i.test(f.error) && <p className="auth-lead auth-lead--note">{admin ? (allowSignup ? 'No administrator exists yet, so there is nothing to sign in to. Open "Set up admin" to create it.' : 'Use the user name you chose when the administrator account was created.') : 'New here? Open "Create account" first. Otherwise check your user name and password.'}</p>}
    <button className="button button--primary button--wide" type="submit" disabled={f.busy}>{f.busy ? 'One moment…' : f.mode === 'signin' ? (admin ? 'Enter console' : 'Sign in') : 'Create account'} {!f.busy && <Icon name="arrow" size={17} />}</button>
    <button type="button" className="text-button auth-switch" onClick={() => go(admin ? '/login/caller' : '/login/admin')}>{admin ? 'I am a caller instead →' : 'I am an admin →'}</button>
  </form>
}

function AdminLogin({ go, onLogin }) {
  const [setup, setSetup] = useState(null)                         // null = checking, true = no administrator yet
  const flow = useAccountFlow('admin', onLogin, 'signin')
  const { setMode } = flow
  useEffect(() => {
    let live = true
    api('/api/auth/status').then(response => response.json()).then(data => { if (live) { setSetup(!data.admin_exists); if (!data.admin_exists) setMode('signup') } }).catch(() => { if (live) setSetup(false) })
    return () => { live = false }
  }, [setMode])
  return <div className="auth auth--admin page-transition">
    <aside className="auth-side"><AuthBrand light /><div className="auth-side__scene"><SignalScene size={320}><Orb compact /></SignalScene></div><div className="auth-side__copy"><div className="eyebrow"><span className="eyebrow-line" />ADMIN CONSOLE</div><h2>Shape the conversation.</h2><ul><li><Icon name="grid" size={16} />Create and edit forms in the studio</li><li><Icon name="check" size={16} />Publish when your team is ready</li><li><Icon name="pulse" size={16} />Review every response and outcome</li></ul></div></aside>
    <main className="auth-panel"><AccountForm flow={flow} role="admin" allowSignup={setup === true} go={go} setupNote={setup === true ? 'No administrator account exists yet. Open "Set up admin" to create it.' : ''} /></main></div>
}

function CallerLogin({ go, onLogin }) {
  const flow = useAccountFlow('user', onLogin, 'signin')
  return <div className="auth auth--caller page-transition"><div className="auth-orbs"><i /><i /><i /></div>
    <AuthBrand />
    <main className="caller-login">
      <AccountForm flow={flow} role="user" allowSignup go={go} />
      <div className="caller-login__scene"><SignalScene size={340}><Orb compact /></SignalScene><span className="caller-login__hint">Drag the ring</span></div>
    </main></div>
}

function App() {
  const clean = value => value.replace(/\/+$/, '') || '/'
  const [session, setSession] = useState(loadSession)
  const [path, setPath] = useState(() => clean(window.location.pathname))
  const [curtain, setCurtain] = useState('')
  const [forms, setForms] = useState(FALLBACK_FORMS)
  const [calls, setCalls] = useState(FALLBACK_CALLS)
  const [backendConnected, setBackendConnected] = useState(false)
  const [setup, setSetup] = useState(false)
  const [selectedForm, setSelectedForm] = useState(FALLBACK_FORMS[0])
  const [language, setLanguage] = useState('en')
  const [call, setCall] = useState(null)
  const [detail, setDetail] = useState(null)
  const [editor, setEditor] = useState(undefined)
  const [upload, setUpload] = useState(false)
  const [toDelete, setToDelete] = useState(null)
  const [accessFor, setAccessFor] = useState(null)
  const [deleting, setDeleting] = useState(false)
  const [deleteError, setDeleteError] = useState('')
  const [motionOff, setMotionOffState] = useState(() => store.get('svara.motion') === 'off')

  const role = session?.role
  const home = role === 'admin' ? '/admin' : '/caller'
  const navigate = useCallback((to, replace = false) => {
    if (window.location.pathname !== to) window.history[replace ? 'replaceState' : 'pushState']({}, '', to)
    setPath(clean(to))
  }, [])
  const withCurtain = useCallback(action => {
    setCurtain('cover')
    setTimeout(() => { action(); window.scrollTo(0, 0); setCurtain('out'); setTimeout(() => setCurtain(''), 650) }, 520)
  }, [])

  // route guard: signed-out visitors only see the sign-in pages; each role only sees its own pages
  useEffect(() => {
    if (!session) { if (!path.startsWith('/login')) navigate('/login', true); return }
    const wrongArea = path.startsWith('/login') || path === '/' || (role === 'user' && (path === '/admin' || path === '/responses' || path === '/settings')) || (role === 'admin' && path === '/caller')
    if (wrongArea) navigate(home, true)
  }, [session, path, role, home, navigate])

  useEffect(() => {
    const onPopState = () => setPath(clean(window.location.pathname))
    window.addEventListener('popstate', onPopState)
    return () => window.removeEventListener('popstate', onPopState)
  }, [])

  useEffect(() => {
    onSessionExpired = () => { saveSession(null); setSession(null); setCall(null); navigate('/login', true) }
    if (session) api('/api/auth/me').catch(() => {})                       // an old tab: find out now if the server still knows this session
    return () => { onSessionExpired = null }
  }, [])

  const login = useCallback(next => withCurtain(() => { saveSession(next); setSession(next); navigate(next.role === 'admin' ? '/admin' : '/caller', true) }), [withCurtain, navigate])
  const logout = useCallback(() => withCurtain(() => { api('/api/auth/logout', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' }).catch(() => {}); saveSession(null); setSession(null); setCall(null); setSetup(false); setEditor(undefined); setDetail(null); setUpload(false); setToDelete(null); setAccessFor(null); navigate('/login', true) }), [withCurtain, navigate])

  const view = (() => { const wanted = path === '/forms' ? 'forms' : path === '/responses' ? 'responses' : path === '/settings' ? 'settings' : 'overview'; return role === 'user' && (wanted === 'responses' || wanted === 'settings') ? 'overview' : wanted })()
  const setView = useCallback(next => { navigate(next === 'overview' ? home : `/${next}`); window.scrollTo({ top: 0 }) }, [navigate, home])

  useEffect(() => {
    if (!session) return undefined
    let live = true
    const admin = session.role === 'admin'
    Promise.all([api('/api/forms').then(response => response.json()), admin ? api('/api/calls').then(response => response.json()) : Promise.resolve(null)]).then(([fetchedForms, nextCalls]) => {
      if (!live) return
      let nextForms = fetchedForms
      if (Array.isArray(nextForms)) { nextForms = decorateForms(nextForms); setForms(nextForms); setSelectedForm(nextForms[0]); setLanguage(nextForms[0]?.language || 'en') }
      if (Array.isArray(nextCalls)) setCalls(nextCalls.map(normalizeCall))
      setBackendConnected(true)
    }).catch(() => setBackendConnected(false))
    return () => { live = false }
  }, [session?.role])

  // scroll effects: progress bar, parallax variable, and reveal-on-scroll for cards and sections
  useEffect(() => {
    const root = document.documentElement
    const onScroll = () => { const max = root.scrollHeight - root.clientHeight; root.style.setProperty('--scroll', String(Math.round(window.scrollY))); root.style.setProperty('--progress', max > 0 ? String(window.scrollY / max) : '0') }
    onScroll(); window.addEventListener('scroll', onScroll, { passive: true }); window.addEventListener('resize', onScroll)
    return () => { window.removeEventListener('scroll', onScroll); window.removeEventListener('resize', onScroll) }
  }, [])
  useLayoutEffect(() => {
    const nodes = [...document.querySelectorAll('.hero-card,.stat-card,.step-card,.section-heading,.form-card,.recent-row,.create-card,.forms-intro,.responses-head,.response-summary,.table-card,.library-toolbar')]
    nodes.forEach((node, index) => { if (!node.classList.contains('reveal')) { node.classList.add('reveal'); node.style.setProperty('--i', String(index % 5)) } })
    const observer = new IntersectionObserver(entries => entries.forEach(entry => { if (entry.isIntersecting) { entry.target.classList.add('is-in'); observer.unobserve(entry.target) } }), { threshold: 0.1 })
    nodes.forEach(node => { if (!node.classList.contains('is-in')) observer.observe(node) })
    return () => observer.disconnect()
  }, [view, role, forms, calls])

  const setMotionOff = value => { setMotionOffState(value); store.set('svara.motion', value ? 'off' : 'on') }
  useEffect(() => { document.documentElement.dataset.motion = motionOff ? 'off' : 'on' }, [motionOff])
  const renameUser = name => { const next = { ...session, name }; saveSession(next); setSession(next) }
  const updateSettings = (id, settings) => setForms(current => current.map(item => item.id === id ? { ...item, settings } : item))
  const askDelete = form => { setDeleteError(''); setToDelete(form) }
  const confirmDelete = async () => {
    if (!toDelete.mine && !toDelete.draft && forms.filter(item => !item.mine).length <= 1) { setDeleteError('Keep at least one form: callers need something to fill in.'); return }
    setDeleting(true); setDeleteError('')
    try {
      if (backendConnected && !toDelete.draft) await jsonApi('/api/admin/forms/delete', { id: toDelete.id })
      const remaining = forms.filter(item => item.id !== toDelete.id)
      setForms(remaining); setSelectedForm(current => current?.id === toDelete.id ? remaining[0] : current); setToDelete(null)
    } catch (caught) { setDeleteError(caught.message || 'The form could not be deleted.') } finally { setDeleting(false) }
  }

  const unlockByCode = async code => {
    const result = await jsonApi('/api/forms/unlock-by-code', { code })
    const list = await api('/api/forms').then(response => response.json())
    const next = decorateForms(list, forms)
    setForms(next)
    return result.forms.map(item => next.find(form => form.id === item.id)).filter(Boolean)
  }
  const accessChanged = (id, code) => setForms(current => current.map(item => item.id === id ? { ...item, access_code: code, has_code: Boolean(code) } : item))

  const openSetup = form => {
    const open = forms.filter(item => !item.locked)
    const target = form || (selectedForm && forms.some(item => item.id === selectedForm.id) ? selectedForm : open[0])
    if (!target) { setView('forms'); return }                       // no form yet: the caller needs an access code first
    setSelectedForm(target); setLanguage(target?.language || target?.languages?.[0] || 'en'); setSetup(true) }
  const startCall = form => { setSetup(false); setCall({ form: form || selectedForm, language }) }
  const placeForm = (saved, oldId) => { setForms(current => current.some(item => item.id === oldId) ? current.map(item => item.id === oldId ? saved : item) : [saved, ...current.filter(item => item.id !== saved.id)]); setSelectedForm(current => current?.id === oldId ? saved : current) }
  const saveForm = async (nextForm, published) => {
    let saved = { ...nextForm, draft: !published }
    if (published && backendConnected) {
      const response = await jsonApi('/api/admin/forms/save', { id: nextForm.id, title: nextForm.title, languages: nextForm.languages, fields: nextForm.fields.map(field => ({ name: field.name, label: field.label, question: field.question, kind: field.kind, required: field.required })) })
      saved = { ...nextForm, ...response, draft: false }
    } else if (published) saved = { ...nextForm, draft: false }
    placeForm(saved, nextForm.id); setEditor(undefined)
  }
  const onUploaded = async saved => {
    try { const list = await api('/api/forms').then(response => response.json()); setForms(current => decorateForms(list, current)) } catch { placeForm(decorateForms([saved])[0], saved.id) }
  }

  const overlay = <><div className="scroll-progress" aria-hidden="true" />{curtain && <div className={`curtain curtain--${curtain}`} aria-hidden="true"><BrandMark /></div>}</>

  if (!session) {
    const route = path === '/login/admin' ? 'admin' : path === '/login/caller' ? 'caller' : 'chooser'
    const go = to => navigate(to)
    return <>{overlay}{route === 'admin' ? <AdminLogin key="admin" go={go} onLogin={login} /> : route === 'caller' ? <CallerLogin key="caller" go={go} onLogin={login} /> : <RoleChooser key="chooser" go={go} />}</>
  }

  const admin = role === 'admin'
  return <div className="app-shell">{overlay}<Sidebar session={session} view={view} setView={setView} onLogout={logout} /><div className="main-shell"><Topbar role={role} view={view} name={session.name} onStart={() => openSetup()} onLogout={logout} onUpload={() => setUpload(true)} backendConnected={backendConnected} /><div className="content-shell"><div className="page-transition" key={`${role}-${view}`}>{view === 'overview' && <Overview forms={forms} calls={calls} role={role} onEdit={setEditor} onDelete={askDelete} onAccess={setAccessFor} onUnlockCode={unlockByCode} onStart={openSetup} setView={setView} />}{view === 'forms' && <FormsView forms={forms} role={role} onEdit={setEditor} onDelete={askDelete} onAccess={setAccessFor} onUnlockCode={unlockByCode} onCreate={() => setEditor(null)} onUpload={() => setUpload(true)} onStart={openSetup} />}{view === 'settings' && admin && <SettingsView session={session} forms={forms} connected={backendConnected} onRename={renameUser} motionOff={motionOff} setMotionOff={setMotionOff} onSettings={updateSettings} onLogout={logout} />}{view === 'responses' && admin && <ResponsesView calls={calls} onSelect={setDetail} />}</div></div></div><MobileNav role={role} view={view} setView={setView} />{setup && <CallSetup form={selectedForm} setForm={setSelectedForm} forms={forms.filter(item => !item.locked)} language={language} setLanguage={setLanguage} onStart={startCall} onClose={() => setSetup(false)} />}{admin && editor !== undefined && <FormEditor form={editor} onClose={() => setEditor(undefined)} onSave={saveForm} />}{call && <CallModal form={call.form} language={call.language} initials={initials(session.name)} onClose={() => { setCall(null); setView('overview') }} onFinished={() => {}} />}{admin && accessFor && <AccessDialog form={accessFor} onClose={() => setAccessFor(null)} onChanged={accessChanged} />}{toDelete && <ConfirmDialog title={`Delete “${toDelete.title}”?`} body={toDelete.draft ? 'This draft will be removed.' : toDelete.mine ? 'Your uploaded form and its PDF will be removed. Calls you already made are kept.' : 'Callers will no longer see this form, and its uploaded PDF is removed. Calls already recorded are kept.'} confirmLabel="Delete form" busy={deleting} error={deleteError} onConfirm={confirmDelete} onCancel={() => setToDelete(null)} />}{upload && <UploadFormModal personal={!admin} connected={backendConnected} onClose={() => setUpload(false)} onPublished={onUploaded} />}{admin && detail && <CallDetail call={detail} connected={backendConnected} onClose={() => setDetail(null)} />}</div>
}

createRoot(document.getElementById('root')).render(<ErrorBoundary><App /></ErrorBoundary>)
