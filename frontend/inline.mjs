// Inlines the built JS/CSS/SVG into one HTML file, because the call server serves a single page with a strict CSP.
import fs from 'node:fs'
const d = 'dist', htmlPath = `${d}/index.html`
let html = fs.readFileSync(htmlPath, 'utf8')
html = html.replace(/<link rel="stylesheet"[^>]*href="([^"]+)"[^>]*>/g, (_, h) => `<style>${fs.readFileSync(d + h, 'utf8')}</style>`)
html = html.replace(/<script type="module"[^>]*src="([^"]+)"[^>]*><\/script>/g, (_, s) => `<script type="module">${fs.readFileSync(d + s, 'utf8').split('__TOKEN__').join('\\x5f_TOKEN__').replace(/<\/script/g, '<\\/script')}</script>`)
const svg = fs.readFileSync(`${d}/svara-mark.svg`).toString('base64')
html = html.replace(/<link rel="icon"[^>]*>/, `<link rel="icon" href="data:image/svg+xml;base64,${svg}" />`)
fs.writeFileSync('../callpage/static/index.html', html.split('/svara-mark.svg').join(`data:image/svg+xml;base64,${svg}`))
console.log('wrote ../callpage/static/index.html', html.length, 'bytes')
