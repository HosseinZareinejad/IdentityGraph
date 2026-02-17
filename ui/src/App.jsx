import { useState } from 'react'

function App() {
  const [formData, setFormData] = useState({
    full_name: '',
    username: '',
    email: '',
    phone_number: '',
    city: '',
    bio: '',
    birth_year: ''
  })
  
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState([])
  const [error, setError] = useState(null)

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value })
  }

  const handleSearch = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    
    try {
      const payload = {
        ...formData,
        birth_year: formData.birth_year ? parseInt(formData.birth_year) : null
      }
      
      const response = await fetch('http://127.0.0.1:8000/resolve', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload)
      })
      
      if (!response.ok) {
        throw new Error('خطا در برقراری ارتباط با سرور')
      }
      
      const data = await response.json()
      setResults(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen p-8 max-w-7xl mx-auto">
      <header className="mb-10 text-center">
        <h1 className="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-accent mb-2">
          موتور تطبیق هویت VectorID
        </h1>
        <p className="text-gray-400">جستجوی هوشمند و شناسایی پروفایل‌های مشابه با استفاده از هوش مصنوعی</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        {/* Search Form Panel */}
        <div className="lg:col-span-5 bg-surface p-6 rounded-2xl shadow-xl border border-gray-800 h-fit">
          <h2 className="text-xl font-bold mb-6 flex items-center gap-2">
            <span className="bg-primary p-2 rounded-lg text-white">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M8 4a4 4 0 100 8 4 4 0 000-8zM2 8a6 6 0 1110.89 3.476l4.817 4.817a1 1 0 01-1.414 1.414l-4.816-4.816A6 6 0 012 8z" clipRule="evenodd" />
              </svg>
            </span>
            مشخصات هدف
          </h2>
          
          <form onSubmit={handleSearch} className="space-y-4">
            <div>
              <label className="block text-sm text-gray-400 mb-1">نام و نام خانوادگی</label>
              <input type="text" name="full_name" value={formData.full_name} onChange={handleChange} required
                className="w-full bg-background border border-gray-700 rounded-lg p-3 text-white focus:ring-2 focus:ring-primary focus:border-transparent outline-none transition-all"
                placeholder="مثال: علی حسینی" />
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm text-gray-400 mb-1">نام کاربری</label>
                <input type="text" name="username" value={formData.username} onChange={handleChange}
                  className="w-full bg-background border border-gray-700 rounded-lg p-3 text-white focus:ring-2 focus:ring-primary focus:border-transparent outline-none transition-all" />
              </div>
              <div>
                <label className="block text-sm text-gray-400 mb-1">ایمیل</label>
                <input type="email" name="email" value={formData.email} onChange={handleChange}
                  className="w-full bg-background border border-gray-700 rounded-lg p-3 text-white focus:ring-2 focus:ring-primary focus:border-transparent outline-none transition-all" />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm text-gray-400 mb-1">شماره تماس</label>
                <input type="text" name="phone_number" value={formData.phone_number} onChange={handleChange} dir="ltr"
                  className="w-full bg-background border border-gray-700 rounded-lg p-3 text-white focus:ring-2 focus:ring-primary focus:border-transparent outline-none transition-all text-left" />
              </div>
              <div>
                <label className="block text-sm text-gray-400 mb-1">سال تولد</label>
                <input type="number" name="birth_year" value={formData.birth_year} onChange={handleChange}
                  className="w-full bg-background border border-gray-700 rounded-lg p-3 text-white focus:ring-2 focus:ring-primary focus:border-transparent outline-none transition-all" />
              </div>
            </div>

            <div>
              <label className="block text-sm text-gray-400 mb-1">شهر</label>
              <input type="text" name="city" value={formData.city} onChange={handleChange}
                className="w-full bg-background border border-gray-700 rounded-lg p-3 text-white focus:ring-2 focus:ring-primary focus:border-transparent outline-none transition-all" />
            </div>

            <div>
              <label className="block text-sm text-gray-400 mb-1">بیوگرافی (Bio)</label>
              <textarea name="bio" value={formData.bio} onChange={handleChange} rows="3"
                className="w-full bg-background border border-gray-700 rounded-lg p-3 text-white focus:ring-2 focus:ring-primary focus:border-transparent outline-none transition-all resize-none"
                placeholder="توضیحات یا بیوگرافی کاربر..."></textarea>
            </div>

            <button type="submit" disabled={loading}
              className="w-full bg-gradient-to-r from-primary to-accent hover:opacity-90 text-white font-bold py-3 px-4 rounded-lg transition-all flex justify-center items-center gap-2 mt-4 shadow-lg shadow-primary/20">
              {loading ? (
                <span className="animate-spin h-5 w-5 border-2 border-white border-t-transparent rounded-full"></span>
              ) : (
                "جستجو و تطبیق هویت"
              )}
            </button>
          </form>
        </div>

        {/* Results Panel */}
        <div className="lg:col-span-7 bg-surface p-6 rounded-2xl shadow-xl border border-gray-800">
          <h2 className="text-xl font-bold mb-6 flex items-center gap-2">
            <span className="bg-emerald-500 p-2 rounded-lg text-white">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                <path d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z" />
                <path fillRule="evenodd" d="M4 5a2 2 0 012-2 3 3 0 003 3h2a3 3 0 003-3 2 2 0 012 2v11a2 2 0 01-2 2H6a2 2 0 01-2-2V5zm3 4a1 1 0 000 2h.01a1 1 0 100-2H7zm3 0a1 1 0 000 2h3a1 1 0 100-2h-3zm-3 4a1 1 0 100 2h.01a1 1 0 100-2H7zm3 0a1 1 0 100 2h3a1 1 0 100-2h-3z" clipRule="evenodd" />
              </svg>
            </span>
            کاندیداهای یافت شده
            <span className="mr-auto text-sm bg-background px-3 py-1 rounded-full text-gray-400 border border-gray-700">
              {results.length} مورد
            </span>
          </h2>

          {error && (
            <div className="bg-red-500/10 border border-red-500/50 text-red-400 p-4 rounded-lg mb-4">
              {error}
            </div>
          )}

          {!loading && results.length === 0 && !error && (
            <div className="flex flex-col items-center justify-center h-64 text-gray-500">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 mb-4 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
              </svg>
              <p>هنوز جستجویی انجام نشده است.</p>
            </div>
          )}

          <div className="space-y-4">
            {results.map((cand, idx) => (
              <div key={cand.record_id} 
                   className="bg-background border border-gray-700 p-5 rounded-xl hover:border-gray-500 transition-all duration-300 transform hover:-translate-y-1 group">
                
                <div className="flex justify-between items-start mb-3">
                  <div>
                    <h3 className="font-bold text-lg text-blue-100 flex items-center gap-2">
                      {cand.profile.full_name}
                      {cand.profile.is_noisy && (
                        <span className="text-[10px] bg-amber-500/20 text-amber-400 px-2 py-0.5 rounded-full border border-amber-500/30">
                          نویز دار
                        </span>
                      )}
                    </h3>
                    <p className="text-xs text-gray-400 mt-1 font-mono">ID: {cand.record_id}</p>
                  </div>
                  
                  {/* Score Circle/Badge */}
                  <div className="flex flex-col items-end">
                    <span className={`text-xl font-black ${cand.score >= 0.9 ? 'text-emerald-400' : cand.score >= 0.75 ? 'text-blue-400' : 'text-amber-400'}`}>
                      {(cand.score * 100).toFixed(1)}%
                    </span>
                    <span className="text-[10px] text-gray-500">میزان انطباق</span>
                  </div>
                </div>

                {/* Score Progress Bar */}
                <div className="w-full bg-gray-800 rounded-full h-1.5 mb-4 overflow-hidden">
                  <div 
                    className={`h-1.5 rounded-full ${cand.score >= 0.9 ? 'bg-emerald-500' : cand.score >= 0.75 ? 'bg-blue-500' : 'bg-amber-500'}`}
                    style={{ width: `${cand.score * 100}%` }}>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-y-2 gap-x-4 text-sm">
                  <div className="flex items-center text-gray-300">
                    <span className="text-gray-500 w-20">نام کاربری:</span>
                    <span className="truncate" dir="ltr">{cand.profile.username || '-'}</span>
                  </div>
                  <div className="flex items-center text-gray-300">
                    <span className="text-gray-500 w-20">شماره:</span>
                    <span className="truncate" dir="ltr">{cand.profile.phone_number || '-'}</span>
                  </div>
                  <div className="flex items-center text-gray-300">
                    <span className="text-gray-500 w-20">ایمیل:</span>
                    <span className="truncate" dir="ltr">{cand.profile.email || '-'}</span>
                  </div>
                  <div className="flex items-center text-gray-300">
                    <span className="text-gray-500 w-20">شهر:</span>
                    <span className="truncate">{cand.profile.city || '-'}</span>
                  </div>
                </div>
                
                {cand.profile.bio && (
                  <div className="mt-3 text-sm bg-gray-800/50 p-3 rounded-lg text-gray-400 line-clamp-2">
                    {cand.profile.bio}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
