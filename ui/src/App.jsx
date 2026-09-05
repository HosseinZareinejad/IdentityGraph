import { useState, useEffect } from 'react'

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8500'

const PLATFORM_STYLE = {
  twitter: { label: 'Twitter', color: 'bg-sky-500/15 text-sky-300 border-sky-500/30' },
  instagram: { label: 'Instagram', color: 'bg-pink-500/15 text-pink-300 border-pink-500/30' },
  telegram: { label: 'Telegram', color: 'bg-cyan-500/15 text-cyan-300 border-cyan-500/30' },
  linkedin: { label: 'LinkedIn', color: 'bg-blue-500/15 text-blue-300 border-blue-500/30' },
}

function PlatformBadge({ platform }) {
  const s = PLATFORM_STYLE[platform] || { label: platform, color: 'bg-slate-500/15 text-slate-300 border-slate-500/30' }
  return <span className={`px-2 py-0.5 rounded-md text-xs font-medium border ${s.color}`}>{s.label}</span>
}

/* A confidence number that attributes an account to a real person should never
   be shown as a bare percentage with no sense of how much to trust it. */
function Confidence({ value, label }) {
  const pct = value == null ? null : Math.round(value * 100)
  const tone = pct == null ? 'text-slate-400'
    : pct >= 85 ? 'text-emerald-300' : pct >= 60 ? 'text-amber-300' : 'text-rose-300'
  return (
    <div className="text-right">
      <div className={`text-2xl font-semibold tabular-nums ${tone}`}>{pct == null ? '—' : `${pct}%`}</div>
      <div className="text-[11px] uppercase tracking-wide text-slate-500">{label}</div>
    </div>
  )
}

/* The per-feature log-odds breakdown. This is the whole point of using
   Fellegi-Sunter instead of an opaque score: an operator can see that a
   decision rests on an exact phone match rather than on a common name. */
function Evidence({ items }) {
  if (!items || items.length === 0) return <div className="text-sm text-slate-500">No contributing evidence.</div>
  const max = Math.max(...items.map(i => Math.abs(i.weight_bits)), 1)
  return (
    <div className="space-y-1.5">
      {items.map((item, i) => {
        const pos = item.weight_bits >= 0
        const width = `${(Math.abs(item.weight_bits) / max) * 100}%`
        return (
          <div key={i} className="flex items-center gap-3 text-xs">
            <div className="w-44 shrink-0 font-mono text-slate-400">{item.feature}</div>
            <div className="flex-1 h-4 bg-slate-800/60 rounded overflow-hidden flex">
              <div className="w-1/2 flex justify-end">
                {!pos && <div className="h-full bg-rose-500/70 rounded-l" style={{ width }} />}
              </div>
              <div className="w-1/2">
                {pos && <div className="h-full bg-emerald-500/70 rounded-r" style={{ width }} />}
              </div>
            </div>
            <div className={`w-16 text-right tabular-nums ${pos ? 'text-emerald-300' : 'text-rose-300'}`}>
              {item.weight_bits > 0 ? '+' : ''}{item.weight_bits.toFixed(2)}
            </div>
          </div>
        )
      })}
      <div className="text-[11px] text-slate-500 pt-1">
        Weights are log₂ odds ratios. Missing fields contribute exactly 0 and are omitted.
      </div>
    </div>
  )
}

function FeedbackButtons({ recordId, targetType, targetId, onDone }) {
  const [sent, setSent] = useState(null)
  const send = async (decision) => {
    try {
      await fetch(`${API}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ record_id: recordId, decision, target_type: targetType, target_id: targetId }),
      })
      setSent(decision)
      onDone && onDone()
    } catch { setSent('error') }
  }
  if (sent) {
    return <span className={`text-xs ${sent === 'confirm' ? 'text-emerald-400' : sent === 'reject' ? 'text-rose-400' : 'text-slate-400'}`}>
      {sent === 'confirm' ? '✓ confirmed' : sent === 'reject' ? '✕ rejected' : 'failed'}
    </span>
  }
  return (
    <div className="flex gap-1.5">
      <button onClick={() => send('confirm')}
        className="px-2 py-1 text-xs rounded border border-emerald-600/40 text-emerald-300 hover:bg-emerald-600/15">
        Confirm
      </button>
      <button onClick={() => send('reject')}
        className="px-2 py-1 text-xs rounded border border-rose-600/40 text-rose-300 hover:bg-rose-600/15">
        Reject
      </button>
    </div>
  )
}

function AccountCard({ account }) {
  return (
    <div className="space-y-1">
      <div className="flex items-center gap-2">
        <PlatformBadge platform={account.platform} />
        <span className="font-medium text-slate-100">{account.display_name || '—'}</span>
        <span className="text-slate-500 text-sm font-mono">@{account.username}</span>
      </div>
      {account.bio && <div className="text-sm text-slate-400">{account.bio}</div>}
      <div className="flex flex-wrap gap-x-4 gap-y-0.5 text-xs text-slate-500">
        {account.city && <span>📍 {account.city}</span>}
        {account.birth_year && <span>🎂 {account.birth_year}</span>}
        {account.job_title && <span>💼 {account.job_title}</span>}
        {account.education && <span>🎓 {account.education}</span>}
        {account.phone_number && <span>📞 {account.phone_number}</span>}
        {account.email && <span>✉️ {account.email}</span>}
        <span>{account.num_posts ?? 0} posts</span>
        <span>{account.graph_degree ?? 0} connections</span>
      </div>
    </div>
  )
}

export default function App() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const [dossier, setDossier] = useState(null)
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [expanded, setExpanded] = useState({})

  const refreshStats = () => fetch(`${API}/stats`).then(r => r.json()).then(setStats).catch(() => {})
  useEffect(() => { refreshStats() }, [])

  const search = async (e) => {
    e && e.preventDefault()
    if (!query.trim()) return
    setLoading(true); setError(null); setDossier(null)
    try {
      const res = await fetch(`${API}/search?q=${encodeURIComponent(query)}&limit=15`)
      if (!res.ok) throw new Error(`search failed (${res.status})`)
      setResults((await res.json()).results)
    } catch (err) { setError(err.message) } finally { setLoading(false) }
  }

  const openDossier = async (recordId) => {
    setLoading(true); setError(null)
    try {
      const res = await fetch(`${API}/dossier/by-record/${recordId}`)
      if (!res.ok) throw new Error(`dossier failed (${res.status})`)
      setDossier(await res.json())
      setExpanded({})
    } catch (err) { setError(err.message) } finally { setLoading(false) }
  }

  const ri = dossier?.real_identity

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200">
      <header className="border-b border-slate-800 bg-slate-900/50">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-lg font-semibold text-slate-100">IdentityGraph</h1>
            <p className="text-xs text-slate-500">Cross-platform identity resolution &amp; real-identity mapping</p>
          </div>
          {stats && (
            <div className="flex gap-5 text-xs text-slate-400">
              <div><span className="text-slate-200 font-medium">{stats.accounts.toLocaleString()}</span> accounts</div>
              <div><span className="text-slate-200 font-medium">{stats.clusters.toLocaleString()}</span> identities</div>
              <div><span className="text-slate-200 font-medium">{stats.registry_records.toLocaleString()}</span> registry</div>
              <div><span className="text-slate-200 font-medium">{stats.feedback_labels}</span> labels</div>
            </div>
          )}
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-6 space-y-6">
        <form onSubmit={search} className="flex gap-2">
          <input
            value={query} onChange={e => setQuery(e.target.value)}
            placeholder="Search by display name or username…"
            className="flex-1 bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-sm
                       placeholder:text-slate-600 focus:outline-none focus:border-slate-500"
          />
          <button type="submit" disabled={loading}
            className="px-5 py-2.5 rounded-lg bg-slate-100 text-slate-900 text-sm font-medium
                       hover:bg-white disabled:opacity-50">
            {loading ? '…' : 'Search'}
          </button>
        </form>

        {error && <div className="p-3 rounded-lg border border-rose-800 bg-rose-950/40 text-rose-300 text-sm">{error}</div>}

        {!dossier && results.length > 0 && (
          <div className="border border-slate-800 rounded-lg divide-y divide-slate-800">
            {results.map(r => (
              <button key={r.record_id} onClick={() => openDossier(r.record_id)}
                className="w-full text-left px-4 py-3 hover:bg-slate-900/60 flex items-center gap-3">
                <PlatformBadge platform={r.platform} />
                <span className="font-medium text-slate-100">{r.display_name}</span>
                <span className="text-slate-500 text-sm font-mono">@{r.username}</span>
                <span className="ml-auto text-xs text-slate-500">
                  {r.cluster_size} linked account{r.cluster_size === 1 ? '' : 's'}
                </span>
              </button>
            ))}
          </div>
        )}

        {dossier && (
          <div className="space-y-5">
            <button onClick={() => setDossier(null)} className="text-xs text-slate-500 hover:text-slate-300">
              ← back to results
            </button>

            <section className="border border-slate-800 rounded-lg p-5 bg-slate-900/30">
              <div className="text-[11px] uppercase tracking-wide text-slate-500 mb-2">Query account</div>
              <AccountCard account={dossier.query_account} />
              <div className="mt-3 pt-3 border-t border-slate-800 flex items-center gap-2 text-xs text-slate-500">
                <span>Virtual identity spans:</span>
                {dossier.platforms_covered.map(p => <PlatformBadge key={p} platform={p} />)}
              </div>
            </section>

            <section className="border border-slate-800 rounded-lg overflow-hidden">
              <div className="px-5 py-3 bg-slate-900/50 border-b border-slate-800 flex items-center justify-between">
                <h2 className="text-sm font-medium text-slate-200">
                  Linked accounts ({dossier.linked_accounts.length})
                </h2>
                <span className="text-[11px] text-slate-500">click a row to see why it was linked</span>
              </div>
              {dossier.linked_accounts.length === 0 && (
                <div className="px-5 py-6 text-sm text-slate-500">
                  No other accounts linked — this account stands alone.
                </div>
              )}
              <div className="divide-y divide-slate-800">
                {dossier.linked_accounts.map((link, i) => (
                  <div key={i} className="px-5 py-4">
                    <div className="flex items-start justify-between gap-4">
                      <button className="flex-1 text-left"
                        onClick={() => setExpanded(e => ({ ...e, [i]: !e[i] }))}>
                        <AccountCard account={link.account} />
                      </button>
                      <div className="flex items-center gap-4">
                        <Confidence value={link.match_confidence} label="same person" />
                        <FeedbackButtons
                          recordId={dossier.query_account.record_id}
                          targetType="linked_account"
                          targetId={link.account.record_id}
                          onDone={refreshStats}
                        />
                      </div>
                    </div>
                    {expanded[i] && (
                      <div className="mt-4 pt-4 border-t border-slate-800">
                        <Evidence items={link.evidence} />
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </section>

            <section className="border border-slate-800 rounded-lg overflow-hidden">
              <div className="px-5 py-3 bg-slate-900/50 border-b border-slate-800">
                <h2 className="text-sm font-medium text-slate-200">Real identity</h2>
              </div>
              {!ri ? (
                <div className="px-5 py-6 text-sm text-slate-500">No registry match found.</div>
              ) : (
                <div className="px-5 py-4 space-y-4">
                  <div className="flex items-start justify-between gap-4">
                    <div className="space-y-1">
                      <div className="text-lg font-medium text-slate-100">{ri.full_name}</div>
                      <div className="flex flex-wrap gap-x-4 text-xs text-slate-500">
                        {ri.city && <span>📍 {ri.city}</span>}
                        {ri.birth_year && <span>🎂 {ri.birth_year}</span>}
                        {ri.job && <span>💼 {ri.job}</span>}
                        {ri.education && <span>🎓 {ri.education}</span>}
                      </div>
                      <div className="text-[11px] text-slate-600 font-mono pt-1">{ri.entity_id}</div>
                      {/* A twin pair produces an identical and entirely
                          convincing evidence list for both people, so the
                          evidence panel alone cannot warn about it. This can. */}
                      {ri.contested && (
                        <div className="mt-2 rounded border border-amber-700/50 bg-amber-950/30 px-3 py-2 text-xs text-amber-300">
                          Contested — another registry record fits this evidence almost as
                          well. The evidence supports the pair, not this individual; see the
                          candidates below before acting on it.
                        </div>
                      )}
                    </div>
                    <div className="flex items-center gap-4">
                      <Confidence value={ri.confidence_top1 ?? ri.confidence} label="calibrated" />
                      <FeedbackButtons
                        recordId={dossier.query_account.record_id}
                        targetType="real_identity"
                        targetId={ri.entity_id}
                        onDone={refreshStats}
                      />
                    </div>
                  </div>

                  <div className="pt-3 border-t border-slate-800">
                    <div className="text-[11px] uppercase tracking-wide text-slate-500 mb-2">
                      Why this person
                    </div>
                    <Evidence items={ri.evidence} />
                  </div>

                  {ri.alternatives?.length > 0 && (
                    <div className="pt-3 border-t border-slate-800">
                      <div className="text-[11px] uppercase tracking-wide text-slate-500 mb-2">
                        Other candidates considered
                        <span className="normal-case tracking-normal text-slate-600 ml-2">
                          — share of the evidence, across all candidates
                        </span>
                      </div>
                      {/* Deliberately `share`, not `confidence`. The headline
                          number above is P(the top pick is right); a
                          candidate's own confidence answers a different
                          question and can sit ABOVE it, which reads as though
                          the runner-up won. Shares are one scale, they sum to
                          1 over the candidate list, and a near-tie is visible
                          as one. */}
                      <div className="space-y-1">
                        {[{ ...ri, rank: 0 }, ...ri.alternatives].map((alt, i) => (
                          <div key={i} className="flex items-center justify-between text-sm">
                            <span className={i === 0 ? "text-slate-200" : "text-slate-400"}>
                              {i === 0 && <span className="text-emerald-500 mr-1">▸</span>}
                              {alt.full_name}
                              <span className="text-slate-600 text-xs ml-2">
                                {alt.city} · {alt.birth_year}
                              </span>
                            </span>
                            <span className="text-slate-500 tabular-nums text-xs">
                              {alt.share != null ? `${Math.round(alt.share * 100)}%` : "—"}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </section>
          </div>
        )}
      </main>
    </div>
  )
}
