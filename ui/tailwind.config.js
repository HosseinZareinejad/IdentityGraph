/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: '#3B82F6',
        secondary: '#1F2937',
        accent: '#8B5CF6',
        background: '#0F172A',
        surface: '#1E293B',
      },
      fontFamily: {
        sans: ['Vazirmatn', 'Inter', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
